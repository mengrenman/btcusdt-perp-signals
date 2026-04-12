"""
Live L1 capture from Binance USD-M perpetual futures websocket.

Subscribes to <symbol>@bookTicker streams and logs every best-bid / best-ask
update to parquet, rotating files hourly. This is the true tick-level quote
feed (not a snapshot) — what we need for queue dynamics, spread regimes, and
quote-update intensity features.

Deps:  pip install websockets pyarrow
Run:   python -m src.capture_book_ticker --symbols BTCUSDT ETHUSDT SOLUSDT --hours 24
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import websockets
except ImportError as e:
    raise SystemExit("Missing dep: pip install websockets") from e

ROOT = Path(__file__).resolve().parent.parent / "dataset" / "binance" / "bookTicker_live"
WS_BASE = "wss://fstream.binance.com/stream?streams="

# In-memory buffers, flushed once per hour per symbol.
_buffers: dict[str, deque] = defaultdict(deque)
_stop = False


def _flush(symbol: str, hour_stamp: str) -> None:
    buf = _buffers[symbol]
    if not buf:
        return
    rows = list(buf)
    buf.clear()
    df = pd.DataFrame(rows)
    df["event_time"] = pd.to_datetime(df["event_time"], unit="ms", utc=True)
    df["transaction_time"] = pd.to_datetime(df["transaction_time"], unit="ms", utc=True)
    out_dir = ROOT / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{symbol}-bookTicker-{hour_stamp}.parquet"
    # Append by writing a new file per hour; merge later.
    df.to_parquet(out, index=False)
    print(f"  [flush] {out.name}  ({len(df):,} rows)")


async def _flusher(symbols: list[str]) -> None:
    last_hour = None
    last_beat = time.time()
    last_counts = {s: 0 for s in symbols}
    while not _stop:
        await asyncio.sleep(5)
        now = datetime.now(timezone.utc)
        hour = now.strftime("%Y-%m-%d_%H")
        if last_hour is None:
            last_hour = hour
        if hour != last_hour:
            for s in symbols:
                _flush(s, last_hour)
            last_hour = hour
        # Heartbeat every 30s: show cumulative msgs/sec per symbol.
        if time.time() - last_beat >= 30:
            parts = []
            for s in symbols:
                cur = len(_buffers[s]) + last_counts[s]  # cumulative since start
                rate = (cur - last_counts[s]) / 30.0
                last_counts[s] = cur
                parts.append(f"{s}:{len(_buffers[s]):,}buf/{rate:.0f}hz")
            print(f"  [beat {now.strftime('%H:%M:%S')}] " + " ".join(parts))
            last_beat = time.time()


async def _consume(symbols: list[str]) -> None:
    streams = "/".join(f"{s.lower()}@bookTicker" for s in symbols)
    url = WS_BASE + streams
    backoff = 1
    while not _stop:
        try:
            print(f"[ws] connecting: {url}")
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                backoff = 1
                async for raw in ws:
                    if _stop:
                        break
                    msg = json.loads(raw)
                    d = msg.get("data", msg)
                    # Futures bookTicker payload:
                    # { e:"bookTicker", u, s, b, B, a, A, T, E }
                    sym = d.get("s")
                    if not sym:
                        continue
                    _buffers[sym].append({
                        "update_id": d.get("u"),
                        "symbol": sym,
                        "best_bid_price": float(d["b"]),
                        "best_bid_qty":   float(d["B"]),
                        "best_ask_price": float(d["a"]),
                        "best_ask_qty":   float(d["A"]),
                        "transaction_time": d.get("T"),
                        "event_time":       d.get("E"),
                    })
        except Exception as e:
            print(f"[ws] error: {e}  (reconnect in {backoff}s)")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def main_async(symbols: list[str], hours: float) -> None:
    print(f"Capturing {symbols} -> {ROOT}")
    print(f"Duration: {hours}h  (Ctrl-C to stop early)")

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_stop)

    consumer = asyncio.create_task(_consume(symbols))
    flusher = asyncio.create_task(_flusher(symbols))

    deadline = time.time() + hours * 3600
    while not _stop and time.time() < deadline:
        await asyncio.sleep(1)

    _request_stop()
    await asyncio.sleep(0.5)
    consumer.cancel()
    flusher.cancel()
    for t in (consumer, flusher):
        try:
            await t
        except asyncio.CancelledError:
            pass

    # Final flush for whatever is still buffered.
    final = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H")
    for s in symbols:
        _flush(s, final)
    print("Done.")


def _request_stop() -> None:
    global _stop
    _stop = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--hours", type=float, default=24.0)
    args = ap.parse_args()
    asyncio.run(main_async(args.symbols, args.hours))


if __name__ == "__main__":
    main()
