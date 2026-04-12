"""
Download Binance USD-M perpetual futures data from data.binance.vision.

No API key required — pulls public bulk archives.

Coverage (per the recovery plan):
  - 12 months of 1-minute klines        (monthly zips)        BTC, ETH, SOL
  - 12 months of funding rate           (monthly zips)        BTC, ETH, SOL
  -  7 days  of aggTrades               (daily zips)          BTC
  -  1 day   of bookTicker (L1 quotes)  (daily zip)           BTC

Output layout:
  dataset/binance/<datatype>/<SYMBOL>/<file>.parquet

Run:  python -m src.download_binance
"""
from __future__ import annotations

import io
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

BASE = "https://data.binance.vision/data/futures/um"
ROOT = Path(__file__).resolve().parent.parent / "dataset" / "binance"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
AGG_COLS = [
    "agg_id", "price", "qty", "first_id", "last_id",
    "timestamp", "is_buyer_maker",
]
BOOK_COLS = [
    "update_id", "best_bid_price", "best_bid_qty",
    "best_ask_price", "best_ask_qty", "transaction_time", "event_time",
]
FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


def _fetch(url: str) -> bytes | None:
    """GET a URL, return body bytes or None on 404."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=60) as r:
            return r.read()
    except Exception as e:
        print(f"  ! {url} -> {e}")
        return None


def _read_zip_csv(blob: bytes, names: list[str]) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            inner = z.namelist()[0]
            with z.open(inner) as f:
                # Binance sometimes ships a header row, sometimes not — sniff it.
                head = f.read(256).decode("utf-8", errors="ignore")
            with z.open(inner) as f:
                has_header = any(c.isalpha() for c in head.split(",")[0])
                df = pd.read_csv(
                    f,
                    header=0 if has_header else None,
                    names=None if has_header else names,
                )
        # Normalize column count if header was present but differs
        if has_header and len(df.columns) == len(names):
            df.columns = names
        return df
    except Exception as e:
        print(f"  ! unzip/parse failed: {e}")
        return None


def _save(df: pd.DataFrame, datatype: str, symbol: str, stem: str) -> Path:
    out_dir = ROOT / datatype / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stem}.parquet"
    df.to_parquet(out, index=False)
    return out


def _months_back(n: int) -> list[str]:
    """Last n full months as YYYY-MM strings, oldest first, ending last month."""
    today = datetime.now(timezone.utc).date().replace(day=1)
    months = []
    cur = today - timedelta(days=1)  # last day of previous month
    cur = cur.replace(day=1)
    for _ in range(n):
        months.append(cur.strftime("%Y-%m"))
        cur = (cur - timedelta(days=1)).replace(day=1)
    return list(reversed(months))


def _days_back(n: int, end_offset: int = 2) -> list[str]:
    """Last n days as YYYY-MM-DD, oldest first. end_offset avoids today (not yet published)."""
    today = datetime.now(timezone.utc).date() - timedelta(days=end_offset)
    return [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n - 1, -1, -1)]


# ---------------------------------------------------------------------------
# Downloaders
# ---------------------------------------------------------------------------
def download_klines(symbol: str, months: list[str], interval: str = "1m") -> None:
    print(f"[klines] {symbol} {interval}  ({len(months)} months)")
    for ym in months:
        stem = f"{symbol}-{interval}-{ym}"
        out = ROOT / "klines" / symbol / f"{stem}.parquet"
        if out.exists():
            print(f"  . {stem}  (cached)")
            continue
        url = f"{BASE}/monthly/klines/{symbol}/{interval}/{stem}.zip"
        blob = _fetch(url)
        if blob is None:
            continue
        df = _read_zip_csv(blob, KLINE_COLS)
        if df is None or df.empty:
            continue
        for c in ["open", "high", "low", "close", "volume",
                  "quote_volume", "taker_buy_base", "taker_buy_quote"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df = df.drop(columns=["ignore"])
        path = _save(df, "klines", symbol, stem)
        print(f"  + {path.name}  ({len(df):,} rows)")


def download_funding(symbol: str, months: list[str]) -> None:
    print(f"[funding] {symbol}  ({len(months)} months)")
    for ym in months:
        stem = f"{symbol}-fundingRate-{ym}"
        out = ROOT / "fundingRate" / symbol / f"{stem}.parquet"
        if out.exists():
            print(f"  . {stem}  (cached)")
            continue
        url = f"{BASE}/monthly/fundingRate/{symbol}/{stem}.zip"
        blob = _fetch(url)
        if blob is None:
            continue
        df = _read_zip_csv(blob, FUNDING_COLS)
        if df is None or df.empty:
            continue
        df["calc_time"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
        df["last_funding_rate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")
        path = _save(df, "fundingRate", symbol, stem)
        print(f"  + {path.name}  ({len(df):,} rows)")


def download_agg_trades(symbol: str, days: list[str]) -> None:
    print(f"[aggTrades] {symbol}  ({len(days)} days)")
    for d in days:
        stem = f"{symbol}-aggTrades-{d}"
        out = ROOT / "aggTrades" / symbol / f"{stem}.parquet"
        if out.exists():
            print(f"  . {stem}  (cached)")
            continue
        url = f"{BASE}/daily/aggTrades/{symbol}/{stem}.zip"
        blob = _fetch(url)
        if blob is None:
            continue
        df = _read_zip_csv(blob, AGG_COLS)
        if df is None or df.empty:
            continue
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
        path = _save(df, "aggTrades", symbol, stem)
        print(f"  + {path.name}  ({len(df):,} rows)")


def download_book_depth(symbol: str, days: list[str]) -> None:
    """Top-of-book depth snapshots (~100ms cadence, 5 levels)."""
    print(f"[bookDepth] {symbol}  ({len(days)} days)")
    for d in days:
        stem = f"{symbol}-bookDepth-{d}"
        out = ROOT / "bookDepth" / symbol / f"{stem}.parquet"
        if out.exists():
            print(f"  . {stem}  (cached)")
            continue
        url = f"{BASE}/daily/bookDepth/{symbol}/{stem}.zip"
        blob = _fetch(url)
        if blob is None:
            continue
        # bookDepth ships with a header row; let pandas infer columns.
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f)
        except Exception as e:
            print(f"  ! parse failed: {e}")
            continue
        if df.empty:
            continue
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        path = _save(df, "bookDepth", symbol, stem)
        print(f"  + {path.name}  ({len(df):,} rows)")


def download_book_ticker(symbol: str, periods: list[str], freq: str = "daily") -> None:
    print(f"[bookTicker] {symbol}  ({len(periods)} {freq})")
    for p in periods:
        stem = f"{symbol}-bookTicker-{p}"
        out = ROOT / "bookTicker" / symbol / f"{stem}.parquet"
        if out.exists():
            print(f"  . {stem}  (cached)")
            continue
        url = f"{BASE}/{freq}/bookTicker/{symbol}/{stem}.zip"
        blob = _fetch(url)
        if blob is None:
            continue
        df = _read_zip_csv(blob, BOOK_COLS)
        if df is None or df.empty:
            continue
        for c in ["best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["transaction_time"] = pd.to_datetime(df["transaction_time"], unit="ms", utc=True)
        df["event_time"] = pd.to_datetime(df["event_time"], unit="ms", utc=True)
        path = _save(df, "bookTicker", symbol, stem)
        print(f"  + {path.name}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    months = _months_back(12)
    week = _days_back(7)
    one_day = _days_back(1)

    print(f"Months: {months[0]} .. {months[-1]}")
    print(f"Week:   {week[0]} .. {week[-1]}")
    print(f"L1 day: {one_day[0]}")
    print(f"Output: {ROOT}\n")

    for sym in SYMBOLS:
        download_klines(sym, months, interval="1m")
        download_funding(sym, months)

    for sym in SYMBOLS:
        download_agg_trades(sym, week)
    # bookTicker was removed from public futures archives. Use bookDepth
    # (top-5 snapshots) for historical depth context, and capture true
    # bookTicker live via src/capture_book_ticker.py.
    depth_days = _days_back(30, end_offset=14)
    for sym in SYMBOLS:
        download_book_depth(sym, depth_days)

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
