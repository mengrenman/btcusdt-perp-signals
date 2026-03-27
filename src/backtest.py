"""
Simple vectorized backtester for evaluating signal quality.

Assumptions:
- Trade at the close of the signal bar (no look-ahead).
- Position sizing: signal value clipped to [-1, 1] represents fraction of capital.
- Transaction costs: configurable in bps.
- No leverage constraints modeled.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class BacktestResult:
    """Container for backtest results."""
    pnl: pd.Series
    cumulative_pnl: pd.Series
    positions: pd.Series
    turnover: pd.Series
    metrics: dict


def run_backtest(
    bars: pd.DataFrame,
    signal: pd.Series,
    cost_bps: float = 2.0,
    max_position: float = 1.0,
    signal_delay: int = 1,
    rebalance_freq: int = 1,
) -> BacktestResult:
    """
    Vectorized backtest of a signal on OHLCV bars.

    Parameters
    ----------
    bars : DataFrame with 'close' and 'returns' columns
    signal : Series of signal values (positive = long, negative = short)
    cost_bps : round-trip transaction cost in basis points
    max_position : maximum absolute position size
    signal_delay : bars of delay before signal takes effect (1 = trade next bar)
    rebalance_freq : only update position every N bars (reduces turnover)

    Returns
    -------
    BacktestResult with PnL series and summary metrics
    """
    # Position is the lagged, clipped signal
    positions = signal.shift(signal_delay).clip(-max_position, max_position).fillna(0)

    # Subsample positions to rebalance only every N bars
    if rebalance_freq > 1:
        mask = pd.Series(False, index=positions.index)
        mask.iloc[::rebalance_freq] = True
        positions = positions.where(mask).ffill().fillna(0)

    # Forward returns (we enter at close, capture next bar's return)
    fwd_returns = bars["returns"].shift(-1).fillna(0)

    # Gross PnL
    gross_pnl = positions * fwd_returns

    # Transaction costs
    turnover = positions.diff().abs()
    costs = turnover * (cost_bps / 10000)

    # Net PnL
    net_pnl = gross_pnl - costs
    cum_pnl = net_pnl.cumsum()

    # Metrics
    metrics = compute_metrics(net_pnl, positions, turnover)

    return BacktestResult(
        pnl=net_pnl,
        cumulative_pnl=cum_pnl,
        positions=positions,
        turnover=turnover,
        metrics=metrics,
    )


def compute_metrics(pnl: pd.Series, positions: pd.Series, turnover: pd.Series) -> dict:
    """Compute standard trading metrics."""
    n_bars = len(pnl)
    total_pnl = pnl.sum()
    mean_pnl = pnl.mean()
    std_pnl = pnl.std()

    # Sharpe (annualized assuming 1-min bars, 1440 bars/day, 365 days/year)
    bars_per_year = 1440 * 365
    sharpe = (mean_pnl / std_pnl) * np.sqrt(bars_per_year) if std_pnl > 0 else 0

    # Max drawdown
    cum_pnl = pnl.cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()

    # Win rate
    winning = (pnl > 0).sum()
    total_trades = (pnl != 0).sum()
    win_rate = winning / total_trades if total_trades > 0 else 0

    # Profit factor
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Average holding (bars where position is nonzero)
    in_market = (positions.abs() > 0.01).mean()

    # Total turnover
    total_turnover = turnover.sum()

    return {
        "total_pnl": total_pnl,
        "mean_pnl_per_bar": mean_pnl,
        "std_pnl_per_bar": std_pnl,
        "sharpe_annualized": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "in_market_pct": in_market,
        "total_turnover": total_turnover,
        "n_bars": n_bars,
    }


def walk_forward_backtest(
    bars: pd.DataFrame,
    signal_fn,
    train_bars: int = 720,
    test_bars: int = 240,
    cost_bps: float = 2.0,
) -> BacktestResult:
    """
    Walk-forward backtest: train signal parameters on rolling window,
    test on out-of-sample period.

    Parameters
    ----------
    bars : full OHLCV DataFrame
    signal_fn : callable(bars) -> pd.Series of signals
    train_bars : number of bars for training window
    test_bars : number of bars for test window
    cost_bps : transaction cost
    """
    all_pnl = []
    all_positions = []
    all_turnover = []

    start = 0
    while start + train_bars + test_bars <= len(bars):
        # Training window (used to compute signal parameters like rolling stats)
        full_window = bars.iloc[start : start + train_bars + test_bars]
        signal = signal_fn(full_window)

        # Only keep the test portion
        test_signal = signal.iloc[train_bars:]
        test_bars_df = full_window.iloc[train_bars:]

        result = run_backtest(test_bars_df, test_signal, cost_bps=cost_bps)
        all_pnl.append(result.pnl)
        all_positions.append(result.positions)
        all_turnover.append(result.turnover)

        start += test_bars

    if not all_pnl:
        raise ValueError("Not enough data for walk-forward backtest")

    pnl = pd.concat(all_pnl)
    positions = pd.concat(all_positions)
    turnover = pd.concat(all_turnover)
    metrics = compute_metrics(pnl, positions, turnover)

    return BacktestResult(
        pnl=pnl,
        cumulative_pnl=pnl.cumsum(),
        positions=positions,
        turnover=turnover,
        metrics=metrics,
    )
