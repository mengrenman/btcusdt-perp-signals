"""
Data loading and OHLCV bar aggregation for Binance BTCUSDT perpetual futures.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"


def load_raw_trades(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and concatenate all parquet trade files, filtering to clean date boundaries."""
    files = sorted(data_dir.glob("*.parquet"))
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df["timestamp"] = pd.to_datetime(df["trade_time"], unit="ns")
        dfs.append(df)
    trades = pd.concat(dfs, ignore_index=True)
    trades = trades.sort_values("timestamp").reset_index(drop=True)

    # Filter to clean calendar-day boundaries (April 18 and 19)
    trades = trades[
        (trades["timestamp"] >= "2025-04-18") & (trades["timestamp"] < "2025-04-20")
    ].copy()

    # Signed quantity: positive for buyer-initiated (BID), negative for seller-initiated (ASK)
    trades["signed_qty"] = np.where(trades["side"] == "BID", trades["quantity"], -trades["quantity"])
    trades["notional"] = trades["price"] * trades["quantity"]

    return trades


def aggregate_ohlcv(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Aggregate tick trades into OHLCV bars at the given frequency."""
    df = trades.set_index("timestamp")

    bars = df.resample(freq).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("quantity", "sum"),
        notional=("notional", "sum"),
        trade_count=("price", "size"),
        buy_volume=("signed_qty", lambda x: x[x > 0].sum()),
        sell_volume=("signed_qty", lambda x: (-x[x < 0]).sum()),
    )

    bars["vwap"] = bars["notional"] / bars["volume"]
    bars["returns"] = bars["close"].pct_change()
    bars["log_returns"] = np.log(bars["close"] / bars["close"].shift(1))
    bars["spread"] = bars["high"] - bars["low"]
    bars["buy_ratio"] = bars["buy_volume"] / bars["volume"]

    return bars.dropna(subset=["open"])


def compute_microstructure(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Compute per-bar microstructure features from tick data."""
    df = trades.set_index("timestamp")

    def _micro_agg(group):
        if len(group) == 0:
            return pd.Series(dtype=float)
        prices = group["price"]
        quantities = group["quantity"]

        # Kyle's lambda proxy: price impact per unit volume
        price_range = prices.max() - prices.min()
        total_vol = quantities.sum()

        # Trade arrival rate (trades per second)
        if len(group) > 1:
            duration = (group.index[-1] - group.index[0]).total_seconds()
            arrival_rate = len(group) / max(duration, 1e-6)
        else:
            arrival_rate = 0.0

        # Average trade size
        avg_trade_size = quantities.mean()

        # Order flow imbalance (signed)
        ofi = group["signed_qty"].sum()

        return pd.Series({
            "price_impact": price_range / max(total_vol, 1e-10),
            "arrival_rate": arrival_rate,
            "avg_trade_size": avg_trade_size,
            "order_flow_imbalance": ofi,
        })

    micro = df.resample(freq).apply(_micro_agg)
    # Handle potential MultiIndex from apply
    if isinstance(micro.index, pd.MultiIndex):
        micro = micro.unstack(level=-1)

    return micro
