"""
Live paper-trading harness for the BTCUSDT composite short-horizon MR strategy.

Subscribes to Binance USD-M futures websocket for btcusdt@aggTrade (tick trades)
and btcusdt@bookTicker (L1 quotes), aggregates trades into 1-minute bars with
buy/sell split from the aggressor flag, recomputes the composite signal from
src.signals every bar close, determines the hysteresis-thresholded position,
and logs a simulated taker fill at mid +/- half-spread.

Everything is taker-side with no queue modelling. Transaction cost = round-trip
taker fee (bps, default 2.0) charged on every |delta position|; half-spread slip
is applied on top at fill time. Output is hourly parquet files in
dataset/paper_trade/BTCUSDT/.

The composite signal and weights exactly match the strategy described in the
PDF summary sent to Jeremi (2026-04-09): eight sub-signals combined with the
default weights from src.signals.composite_signal.

Deps:  pip install websockets pyarrow pandas
Run:   python -m src.paper_trade
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal as signal_mod
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import websockets
except ImportError as e:
    raise SystemExit("Missing dep: pip install websockets") from e

from src.signals import composite_signal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL = "BTCUSDT"
WS_URL = (
    "wss://fstream.binance.com/stream?streams="
    f"{SYMBOL.lower()}@aggTrade/{SYMBOL.lower()}@bookTicker"
)
ROOT = Path(__file__).resolve().parent.parent / "dataset" / "paper_trade" / SYMBOL

BAR_SECONDS = 60
HISTORY_BARS = 120           # keep 2 hours of bars in memory for rolling signal
COST_BPS_RT = 2.0            # round-trip taker cost (matches the PDF's 2 bps scenario)
WARMUP_BARS = 25             # need at least 25 bars before the 20-lookback signals are valid

_stop = False


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class BarAccumulator:
    """Accumulates trades into a single 1-minute bar."""
    open_time: datetime
    open: float | None = None
    high: float = -np.inf
    low: float = np.inf
    close: float | None = None
    volume: float = 0.0
    notional: float = 0.0
    trade_count: int = 0
    buy_volume: float = 0.0     # aggressor = buyer
    sell_volume: float = 0.0    # aggressor = seller

    def add_trade(self, price: float, qty: float, is_buyer_maker: bool) -> None:
        if self.open is None:
            self.open = price
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += qty
        self.notional += price * qty
        self.trade_count += 1
        # Binance convention: is_buyer_maker=True => the aggressor was a SELLER
        if is_buyer_maker:
            self.sell_volume += qty
        else:
            self.buy_volume += qty

    def to_row(self) -> dict | None:
        if self.trade_count == 0 or self.open is None:
            return None
        return {
            "open_time": self.open_time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "notional": self.notional,
            "trade_count": self.trade_count,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
        }


@dataclass
class BookState:
    best_bid: float | None = None
    best_ask: float | None = None
    ts_ms: int = 0

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def half_spread_bps(self) -> float | None:
        if self.mid is None or self.mid <= 0:
            return None
        return 0.5 * (self.best_ask - self.best_bid) / self.mid * 1e4


@dataclass
class StrategyState:
    bars: deque = field(default_factory=lambda: deque(maxlen=HISTORY_BARS))
    current: BarAccumulator | None = None
    book: BookState = field(default_factory=BookState)
    position: float = 0.0        # current signed position in [-1, +1]
    cash_return: float = 0.0     # cumulative realized log-return minus costs
    bar_log: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bar finalization + signal recompute
# ---------------------------------------------------------------------------
def floor_minute(ts_ms: int) -> datetime:
    ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return ts.replace(second=0, microsecond=0)


def finalize_bar(state: StrategyState) -> dict | None:
    if state.current is None:
        return None
    row = state.current.to_row()
    state.current = None
    if row is None:
        return None

    # Returns column expected by composite signal
    if len(state.bars) > 0 and state.bars[-1]["close"]:
        row["returns"] = np.log(row["close"] / state.bars[-1]["close"])
    else:
        row["returns"] = 0.0

    state.bars.append(row)
    return row


def recompute_signal(state: StrategyState) -> tuple[float, float]:
    """Recompute composite signal and thresholded target position. Returns (composite, target_pos)."""
    if len(state.bars) < WARMUP_BARS:
        return (np.nan, 0.0)
    df = pd.DataFrame(list(state.bars)).set_index("open_time")
    sigs = composite_signal(df)
    comp = float(sigs["composite"].iloc[-1])
    target = float(sigs["position"].iloc[-1])
    if np.isnan(target):
        target = 0.0
    return (comp, target)


def apply_fill(state: StrategyState, target_pos: float, bar_close: float) -> dict:
    """
    Simulate a taker fill that moves position -> target_pos at bar close.
    Pay half-spread + cost_bps/2 on |delta position|. Update PnL accounting.
    """
    old_pos = state.position
    delta = target_pos - old_pos

    # Position P&L: we held old_pos through this bar, and the bar's log-return is bars[-1]['returns']
    bar_return = float(state.bars[-1]["returns"]) if state.bars else 0.0
    pos_pnl = old_pos * bar_return

    # Transaction cost on |delta|: half-spread + half-round-trip fee (in return units)
    hs_bps = state.book.half_spread_bps if state.book.half_spread_bps is not None else 0.0
    # cost applied per unit of |delta|, scaled to return units
    cost_per_unit_bps = hs_bps + (COST_BPS_RT / 2.0)
    txn_cost = abs(delta) * cost_per_unit_bps * 1e-4

    net_pnl = pos_pnl - txn_cost
    state.cash_return += net_pnl
    state.position = target_pos

    fill_price = None
    if delta != 0 and state.book.mid is not None:
        fill_price = state.book.mid + np.sign(delta) * (
            state.book.mid * hs_bps * 1e-4 if hs_bps else 0.0
        )

    return {
        "old_pos": old_pos,
        "new_pos": target_pos,
        "delta": delta,
        "bar_close": bar_close,
        "bar_return": bar_return,
        "pos_pnl": pos_pnl,
        "half_spread_bps": hs_bps,
        "txn_cost": txn_cost,
        "net_pnl": net_pnl,
        "cum_net_return": state.cash_return,
        "fill_price": fill_price,
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def flush(state: StrategyState, hour_stamp: str) -> None:
    if not state.bar_log:
        return
    df = pd.DataFrame(state.bar_log)
    state.bar_log.clear()
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / f"{SYMBOL}-paper-{hour_stamp}.parquet"
    if out.exists():
        # Append by reading and concatenating; file-per-hour is small enough.
        df = pd.concat([pd.read_parquet(out), df], ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"  [flush] {out.name}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# Websocket consumer
# ---------------------------------------------------------------------------
async def consume(state: StrategyState) -> None:
    backoff = 1
    while not _stop:
        try:
            print(f"[ws] connecting: {WS_URL}")
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
                backoff = 1
                async for raw in ws:
                    if _stop:
                        break
                    msg = json.loads(raw)
                    stream = msg.get("stream", "")
                    d = msg.get("data", msg)
                    if "aggTrade" in stream:
                        _on_trade(state, d)
                    elif "bookTicker" in stream:
                        _on_book(state, d)
        except Exception as e:
            print(f"[ws] error: {e}  (reconnect in {backoff}s)")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


def _on_book(state: StrategyState, d: dict) -> None:
    try:
        state.book.best_bid = float(d["b"])
        state.book.best_ask = float(d["a"])
        state.book.ts_ms = int(d.get("T", d.get("E", 0)))
    except (KeyError, ValueError):
        pass


def _on_trade(state: StrategyState, d: dict) -> None:
    try:
        price = float(d["p"])
        qty = float(d["q"])
        ts_ms = int(d["T"])
        is_buyer_maker = bool(d["m"])
    except (KeyError, ValueError):
        return

    bar_minute = floor_minute(ts_ms)

    if state.current is None:
        state.current = BarAccumulator(open_time=bar_minute)
    elif bar_minute > state.current.open_time:
        # Bar rollover: finalize previous bar(s) and act on the new close.
        _close_and_act(state)
        # If there were skipped minutes with no trades, stamp them as empty gaps.
        # We keep it simple and just start the new bar.
        state.current = BarAccumulator(open_time=bar_minute)

    state.current.add_trade(price, qty, is_buyer_maker)


def _close_and_act(state: StrategyState) -> None:
    row = finalize_bar(state)
    if row is None:
        return
    comp, target = recompute_signal(state)
    fill = apply_fill(state, target, bar_close=row["close"])
    entry = {
        "bar_open_time": row["open_time"],
        "close": row["close"],
        "volume": row["volume"],
        "trade_count": row["trade_count"],
        "buy_volume": row["buy_volume"],
        "sell_volume": row["sell_volume"],
        "composite": comp,
        **fill,
        "n_bars_in_memory": len(state.bars),
    }
    state.bar_log.append(entry)


# ---------------------------------------------------------------------------
# Hour rotator / heartbeat
# ---------------------------------------------------------------------------
async def rotator(state: StrategyState) -> None:
    last_hour = None
    last_beat = time.time()
    while not _stop:
        await asyncio.sleep(5)
        now = datetime.now(timezone.utc)
        hour = now.strftime("%Y-%m-%d_%H")
        if last_hour is None:
            last_hour = hour
        if hour != last_hour:
            flush(state, last_hour)
            last_hour = hour
        if time.time() - last_beat >= 60:
            book_str = (
                f"bid={state.book.best_bid} ask={state.book.best_ask}"
                if state.book.best_bid else "book=—"
            )
            print(
                f"  [beat {now.strftime('%H:%M:%S')}] bars={len(state.bars)} "
                f"pos={state.position:+.2f} cum_ret={state.cash_return:+.5f} {book_str}"
            )
            last_beat = time.time()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main_async(hours: float) -> None:
    state = StrategyState()
    print(f"Paper-trading {SYMBOL}  ->  {ROOT}")
    print(f"Composite signal from src.signals.composite_signal (default weights)")
    print(f"Cost model: taker, round-trip fee = {COST_BPS_RT} bps + live half-spread, no queue model")
    print(f"Warmup: {WARMUP_BARS} bars   History: {HISTORY_BARS} bars   Duration: {hours}h")

    loop = asyncio.get_running_loop()
    for sig in (signal_mod.SIGINT, signal_mod.SIGTERM):
        loop.add_signal_handler(sig, _request_stop)

    consumer = asyncio.create_task(consume(state))
    rot = asyncio.create_task(rotator(state))

    deadline = time.time() + hours * 3600
    while not _stop and time.time() < deadline:
        await asyncio.sleep(1)

    _request_stop()
    await asyncio.sleep(0.5)
    consumer.cancel()
    rot.cancel()
    for t in (consumer, rot):
        try:
            await t
        except asyncio.CancelledError:
            pass

    # Final flush
    final_hour = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H")
    flush(state, final_hour)
    print("Done.")


def _request_stop() -> None:
    global _stop
    _stop = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24 * 7, help="max runtime (default: 7 days)")
    args = ap.parse_args()
    asyncio.run(main_async(args.hours))


if __name__ == "__main__":
    main()
