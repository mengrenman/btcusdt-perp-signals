"""
Trading signal generators for 5-15 minute holding period strategies.

All signal functions return a pd.Series aligned to the bar index.
Signal convention: positive = go long, negative = go short.
Signals are normalized to approximately [-1, 1] where possible.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Mean Reversion: Z-score of price vs rolling mean
# ---------------------------------------------------------------------------
def zscore_mean_reversion(bars: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Z-score of close price relative to a rolling window.
    Negative z-score → price below mean → buy signal (mean reversion).
    """
    rolling_mean = bars["close"].rolling(lookback).mean()
    rolling_std = bars["close"].rolling(lookback).std()
    zscore = (bars["close"] - rolling_mean) / rolling_std
    # Invert: low z-score → long signal (mean reversion)
    signal = -zscore
    return signal.rename("zscore_mr")


# ---------------------------------------------------------------------------
# 2. Bollinger Band Mean Reversion
# ---------------------------------------------------------------------------
def bollinger_mean_reversion(bars: pd.DataFrame, lookback: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Position within Bollinger Bands, normalized to [-1, 1].
    Near lower band → +1 (buy), near upper band → -1 (sell).
    """
    rolling_mean = bars["close"].rolling(lookback).mean()
    rolling_std = bars["close"].rolling(lookback).std()
    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std
    # Normalize position within band to [-1, 1]
    bb_position = (bars["close"] - rolling_mean) / (num_std * rolling_std)
    signal = -bb_position.clip(-1, 1)
    return signal.rename("bollinger_mr")


# ---------------------------------------------------------------------------
# 3. VWAP Reversion
# ---------------------------------------------------------------------------
def vwap_reversion(bars: pd.DataFrame, lookback: int = 15) -> pd.Series:
    """
    Mean reversion toward rolling VWAP.
    """
    rolling_notional = bars["notional"].rolling(lookback).sum()
    rolling_volume = bars["volume"].rolling(lookback).sum()
    rolling_vwap = rolling_notional / rolling_volume

    deviation = (bars["close"] - rolling_vwap) / rolling_vwap
    # Standardize
    std = deviation.rolling(lookback).std()
    signal = -(deviation / std)
    return signal.clip(-3, 3).rename("vwap_mr")


# ---------------------------------------------------------------------------
# 4. Order Flow Imbalance
# ---------------------------------------------------------------------------
def order_flow_imbalance(bars: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.Series:
    """
    Smoothed buy/sell volume imbalance — contrarian.
    Excess buying pressure → expect mean-reversion → short signal.
    """
    # Net flow: buy_volume - sell_volume, normalized by total volume
    net_flow = (bars["buy_volume"] - bars["sell_volume"]) / bars["volume"]
    fast_ma = net_flow.rolling(fast).mean()
    slow_ma = net_flow.rolling(slow).mean()
    signal = fast_ma - slow_ma
    # Normalize and invert (contrarian: excess buying → short)
    std = signal.rolling(slow).std()
    signal = -(signal / std).clip(-3, 3)
    return signal.rename("ofi")


# ---------------------------------------------------------------------------
# 5. Trade Intensity Signal
# ---------------------------------------------------------------------------
def trade_intensity(bars: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Abnormal trade count relative to rolling average.
    Spikes in activity combined with price direction give momentum signal.
    """
    avg_count = bars["trade_count"].rolling(lookback).mean()
    std_count = bars["trade_count"].rolling(lookback).std()
    intensity_zscore = (bars["trade_count"] - avg_count) / std_count

    # Combine with return direction: high activity + positive return → long
    ret_sign = np.sign(bars["returns"])
    signal = intensity_zscore * ret_sign
    return signal.clip(-3, 3).rename("trade_intensity")


# ---------------------------------------------------------------------------
# 6. RSI-based Mean Reversion
# ---------------------------------------------------------------------------
def rsi_signal(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    RSI-based signal. Oversold (<30) → buy, overbought (>70) → sell.
    Scaled to [-1, 1].
    """
    delta = bars["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Scale: 50 → 0, 30 → +1, 70 → -1
    signal = -(rsi - 50) / 20
    return signal.clip(-1.5, 1.5).rename("rsi_mr")


# ---------------------------------------------------------------------------
# 7. Dual Moving Average Crossover (Trend/Momentum)
# ---------------------------------------------------------------------------
def ma_crossover(bars: pd.DataFrame, fast: int = 5, slow: int = 15) -> pd.Series:
    """
    Contrarian MA crossover — fade the short-term trend.
    Fast MA above slow → price overextended → short signal.
    """
    fast_ma = bars["close"].ewm(span=fast).mean()
    slow_ma = bars["close"].ewm(span=slow).mean()
    spread = fast_ma - slow_ma
    # Normalize by recent volatility, invert for mean-reversion
    vol = bars["close"].rolling(slow).std()
    signal = -(spread / vol).clip(-3, 3)
    return signal.rename("ma_cross")


# ---------------------------------------------------------------------------
# 8. Volume-Weighted Momentum
# ---------------------------------------------------------------------------
def volume_momentum(bars: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """
    Contrarian volume-weighted momentum — fade high-volume moves.
    Large volume-weighted run-up → expect mean-reversion → short.
    """
    rel_vol = bars["volume"] / bars["volume"].rolling(lookback).mean()
    vol_weighted_ret = bars["returns"] * rel_vol
    signal = vol_weighted_ret.rolling(lookback).sum()
    std = signal.rolling(lookback * 2).std()
    signal = -(signal / std).clip(-3, 3)
    return signal.rename("vol_momentum")


# ---------------------------------------------------------------------------
# Composite Signal
# ---------------------------------------------------------------------------
def composite_signal(bars: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """
    Compute all signals and combine into a weighted composite.
    Returns DataFrame with all individual signals and composite.
    """
    signals = pd.DataFrame(index=bars.index)
    signals["zscore_mr"] = zscore_mean_reversion(bars)
    signals["bollinger_mr"] = bollinger_mean_reversion(bars)
    signals["vwap_mr"] = vwap_reversion(bars)
    signals["ofi"] = order_flow_imbalance(bars)
    signals["trade_intensity"] = trade_intensity(bars)
    signals["rsi_mr"] = rsi_signal(bars)
    signals["ma_cross"] = ma_crossover(bars)
    signals["vol_momentum"] = volume_momentum(bars)

    if weights is None:
        # Default: overweight mean-reversion signals for this regime
        weights = {
            "zscore_mr": 0.15,
            "bollinger_mr": 0.15,
            "vwap_mr": 0.20,
            "ofi": 0.15,
            "trade_intensity": 0.05,
            "rsi_mr": 0.10,
            "ma_cross": 0.10,
            "vol_momentum": 0.10,
        }

    composite = sum(signals[col] * w for col, w in weights.items())
    # Smooth composite to reduce whipsaw/turnover — EMA with ~5-min half-life
    signals["composite_raw"] = composite
    signals["composite"] = composite.ewm(span=5).mean()

    # Thresholded position: only trade when signal is strong enough,
    # and use hysteresis to avoid flipping on noise
    signals["position"] = threshold_position(
        signals["composite"], entry_threshold=0.15, exit_threshold=0.05
    )

    return signals


def threshold_position(
    signal: pd.Series,
    entry_threshold: float = 0.15,
    exit_threshold: float = 0.05,
) -> pd.Series:
    """
    Convert continuous signal to discrete {-1, 0, +1} position with hysteresis.

    Enter long when signal > entry_threshold, exit when it drops below exit_threshold.
    Enter short when signal < -entry_threshold, exit when it rises above -exit_threshold.
    """
    positions = pd.Series(0.0, index=signal.index)
    pos = 0.0

    for i in range(len(signal)):
        val = signal.iloc[i]
        if np.isnan(val):
            positions.iloc[i] = 0.0
            continue

        if pos == 0:
            if val > entry_threshold:
                pos = 1.0
            elif val < -entry_threshold:
                pos = -1.0
        elif pos == 1.0:
            if val < exit_threshold:
                pos = 0.0
            if val < -entry_threshold:
                pos = -1.0
        elif pos == -1.0:
            if val > -exit_threshold:
                pos = 0.0
            if val > entry_threshold:
                pos = 1.0

        positions.iloc[i] = pos

    return positions
