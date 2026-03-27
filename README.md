# BTCUSDT Perpetual Futures: Short-Horizon Signal Research

Quantitative research project developing trading signals for Binance BTCUSDT perpetual futures, targeting a **5–15 minute holding period**. Built from raw tick-level trade data (2 days, ~2.3M trades).

## Project Structure

```
src/
  data.py          Data loading, OHLCV bar aggregation, microstructure features
  signals.py       8 trading signals + weighted composite + hysteresis position sizing
  backtest.py      Vectorized backtester with cost modeling, rebalance control, walk-forward

notebooks/
  01_data_exploration.ipynb   Data quality, return distribution, autocorrelation, intraday patterns
  02_signal_research.ipynb    Signal computation, correlation, IC analysis, quintile decomposition
  03_backtest.ipynb           PnL analysis, cost sensitivity, rebalance sweep, walk-forward validation
```

### Notebooks

1. [Data Exploration & OHLCV Aggregation](notebooks/01_data_exploration.ipynb) — Data quality checks, price/volume overview, return distribution, autocorrelation analysis, intraday patterns
2. [Signal Research & Analysis](notebooks/02_signal_research.ipynb) — Signal computation, correlation matrix, information coefficient across horizons, quintile decomposition
3. [Backtest & Performance Evaluation](notebooks/03_backtest.ipynb) — Gross vs net PnL, transaction cost sensitivity, rebalance frequency sweep, walk-forward validation

```

dataset/
  2025-04-18_67824_trades.parquet   Day 1 tick data
  2025-04-19_67824_trades.parquet   Day 2 tick data
  readme.txt                        Data dictionary
```

## Data Pipeline

Raw trade ticks (nanosecond timestamps, price, quantity, side) are loaded from parquet files, filtered to clean calendar-day boundaries, and aggregated into 1-minute OHLCV bars with additional fields:

- VWAP, notional volume, trade count
- Buy/sell volume split (using trade side: BID = buyer-initiated, ASK = seller-initiated)
- Log returns, spread (high - low), buy ratio

## Signal Suite

All signals are designed for the **mean-reverting microstructure** observed in the data (negative lag-1 return autocorrelation). Signals output values where positive = long, negative = short.

| # | Signal | Type | Description |
|---|--------|------|-------------|
| 1 | Z-score MR | Mean Reversion | Price deviation from rolling mean, normalized by rolling std |
| 2 | Bollinger Band MR | Mean Reversion | Normalized position within Bollinger Bands |
| 3 | VWAP Reversion | Mean Reversion | Deviation from rolling volume-weighted average price |
| 4 | Order Flow Imbalance | Contrarian Microstructure | Fade excess buying/selling pressure (smoothed buy-sell ratio) |
| 5 | Trade Intensity | Microstructure | Abnormal trade count combined with price direction |
| 6 | RSI MR | Mean Reversion | Overbought/oversold reversal via RSI |
| 7 | MA Crossover | Contrarian Momentum | Fade fast/slow EMA spread |
| 8 | Volume Momentum | Contrarian Momentum | Fade volume-weighted cumulative returns |

Signals are combined into a **weighted composite** (overweighting VWAP reversion at 20%), smoothed with a 5-bar EMA to reduce whipsaw, and optionally discretized into {-1, 0, +1} positions via a hysteresis threshold scheme.

## Key Findings

### Signals have real predictive power

All 8 signals show **positive information coefficient** (Spearman rank correlation with 5-min forward returns), ranging from 0.01 to 0.12. The composite achieves IC ~0.10 at the 5-min horizon.

### Gross alpha is meaningful, but thin

| Metric | Value |
|--------|-------|
| Gross annualized Sharpe (1-min rebalance) | ~43 |
| Gross total PnL (2 days) | +0.030 (return units) |
| Breakeven transaction cost | ~0.75 bps round-trip |

### Execution cost is the binding constraint

| Cost (bps) | Net Sharpe | Profitable? |
|------------|-----------|-------------|
| 0 (gross) | +43 | Yes |
| 0.50 | +16 | Yes |
| 0.75 | +3 | Breakeven |
| 1.00 | -10 | No |
| 2.00 | -63 | No |

This is a **maker strategy**: profitable only with limit order execution at sub-1 bps costs, achievable on Binance VIP tiers where makers receive rebates.

### Alpha decays with rebalance frequency

Signal edge is strongest at 1–3 minute rebalancing and decays toward zero by 15 minutes. This means the strategy requires relatively fast execution infrastructure, even though the *conceptual* holding period is 5–15 min.

## How to Run

```bash
# From project root, launch Jupyter
jupyter notebook notebooks/
```

Run the notebooks in order (01 -> 02 -> 03). They import from `src/` via relative path.

### Dependencies

- Python 3.10+
- pandas, numpy, matplotlib, scipy, statsmodels

## Limitations and Next Steps

- **Only 2 days of data** — insufficient for regime-change analysis or seasonal effects.
- **No slippage or queue priority modeling** — critical for real maker strategies.
- **Heuristic signal weights** — could be optimized via LASSO/ridge regression on a longer dataset.
- **No intraday volume normalization** — signals could be scaled by time-of-day volatility.
- Potential additions: funding rate, order book depth/imbalance, cross-asset signals (ETH, SOL correlation).
