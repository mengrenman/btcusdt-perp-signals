# BTCUSDT Perpetual Futures: Short-Horizon Signal Research

Quantitative research on Binance BTCUSDT perpetual futures microstructure and short-horizon trading signals. Started as a 2-day tick-data study of a mean-reverting composite signal; has since been extended with 12 months of klines, 7 days of tick-level `aggTrades` across BTC/ETH/SOL, 30 days of L2 depth snapshots, and a live `bookTicker` websocket capture.

## Project Structure

```
src/
  data.py                   OHLCV aggregation and microstructure features
  signals.py                8 signals + weighted composite + hysteresis position sizing
  backtest.py               Vectorized backtester with cost modeling + walk-forward
  download_binance.py       Bulk puller for data.binance.vision archives
                            (klines, fundingRate, aggTrades, bookDepth)
  capture_book_ticker.py    Live websocket capture of L1 quote updates
  paper_trade.py            Live paper-trading harness (websocket → 1-min bars →
                            composite signal → simulated taker fills → parquet log)

notebooks/
  01_data_exploration.ipynb        Data quality, returns, autocorrelation, intraday
  02_signal_research.ipynb         Signal IC, correlation, quintile decomposition
  03_backtest.ipynb                PnL, cost sensitivity, walk-forward
  05_full_year.ipynb               12-month walk-forward + cross-asset PCA residual MR
  06_dollar_bars.ipynb             Dollar-bar hypothesis test on BTC aggTrades
  07_cross_asset_dollar_bars.ipynb PCA residual MR on BTC-clocked synchronized dollar bars
  08_tick_patterns.ipynb           Tick-level microstructure: sign ACF, power-law fit,
                                   inter-arrival, impact, lead-lag, trade-clock vs wall-clock,
                                   Kyle's lambda, bootstrap CI on gamma, intraday seasonality,
                                   Hasbrouck information share
  09_paper_trade_analysis.ipynb    Live paper-trade log analysis: dedup, PnL recomputation,
                                   equity curve, live-vs-backtest comparison
  10_signal_stacking.ipynb         Signal stacking demo: composite MR + funding carry +
                                   PCA residual, pairwise correlation, √N Sharpe scaling

dataset/
  2025-04-18_67824_trades.parquet   Original 2-day tick sample
  2025-04-19_67824_trades.parquet
  binance/
    klines/{SYM}/      12 months of 1-minute klines, BTC/ETH/SOL
    fundingRate/{SYM}/ 12 months of funding history
    aggTrades/{SYM}/   7 days of tick-level trades (2026-04-01..07)
    bookDepth/{SYM}/   30 days of top-5 L2 snapshots
    bookTicker_live/   Live L1 capture (from capture_book_ticker.py)
  paper_trade/
    BTCUSDT/           Hourly parquet logs from paper_trade.py
```

## Data Pipeline

Two independent sources feed the research:

**Bulk historical archives** (`src/download_binance.py`) pulls public data from
`data.binance.vision`:
- 1-minute klines: 12 months × {BTC, ETH, SOL}
- Funding rate: 12 months × {BTC, ETH, SOL}
- `aggTrades` (tick): 7 days × {BTC, ETH, SOL}, 20M+ prints total
- `bookDepth`: 30 days × {BTC, ETH, SOL} at ~100 ms cadence, 5 levels

**Live L1 capture** (`src/capture_book_ticker.py`) subscribes to
`fstream.binance.com/<symbol>@bookTicker` for true tick-level quote updates,
flushes to parquet hourly. `bookTicker` was removed from Binance's public
futures archives, so live capture is the only way to obtain it.

## Research Threads

### Thread A — Original 2-day composite backtest (NB 01–03)

Eight signals (z-score MR, Bollinger, VWAP reversion, OFI, trade intensity, RSI, contrarian MA crossover, volume momentum) combined into a weighted composite, smoothed with a 5-bar EMA, backtested on 1-minute bars.

- Gross annualized Sharpe ~43 on 2 days (wide CI)
- Breakeven cost ≈ 0.75 bps round-trip
- Unambiguously unprofitable at realistic taker cost (~2 bps)
- Profitable only contingent on maker-side execution

### Thread B — 12-month walk-forward + cross-asset PCA residual MR (NB 05)

Full-year replication of Thread A on 1-minute klines, plus a cross-asset factor
residualization: PC1 of the BTC/ETH/SOL log-return panel explains 84% of
variance and is interpreted as a common market factor. Stripping `β·PC1` from
BTC returns yields a residual with **lag-1 autocorrelation ρ₁ ≈ −0.11**
versus ≈ 0 for raw BTC. The MR signal on the residual has a positive but
sub-bp per-turn edge — below standalone cost.

### Thread C — Dollar-bar hypothesis (NB 06, 07)

Tests whether switching from time bars to **dollar bars** (information clock)
concentrates the signal.

- **NB 06:** BTC-only $5M dollar bars. Edge worse than 1-minute bars (−0.02 bps). Rules out the bar-type hypothesis on single-asset.
- **NB 07:** BTC-clocked synchronized dollar bars with `merge_asof` ETH/SOL joins, rebuilds the PCA residual signal on the dollar-bar clock. Residual autocorrelation *strengthens* to **ρ₁ = −0.13** — the signal is real and cross-sample consistent — but per-turn edge is only **0.086 bps**, matching the $|\rho_1|\sigma^2$ ceiling at this horizon. Robustly sub-bp.

### Thread D — Tick-level microstructure patterns (NB 08)

Direct analysis of 7 days × 3 symbols of `aggTrades`. No strategy, no Sharpe —
just the canonical tick-level diagnostics:

1. **Heavy-tailed trade sizes** (log-log CCDF, power-law tail across 2–3 decades).
2. **Bursty, non-Poisson inter-arrivals** with heavy right tail (Hawkes signature).
3. **Trade-sign persistence (Lo–MacKinlay / Hasbrouck).** In trade-time, $\rho_1 = +0.63$ (BTC), $+0.50$ (ETH). A log-log fit on $k \in [5, 500]$ gives power-law decay exponents $\gamma = 0.78$ (BTC) and $0.99$ (ETH) — squarely inside the equity-literature range $[0.5, 1.0]$. **Crypto perps show the same order-splitting / metaorder signature as equities.**
4. **Two distinct microstructure regimes, cleanly separated by liquidity.** SOL shows the opposite pattern: $\rho_1 = -0.16$ and shallow $\gamma = 0.18$ — the bid-ask-bounce signature of a book where the 271 ms inter-arrival is long enough that consecutive trades alternate across the top of book.
5. **AR(1) on wall-clock signed-flow buckets** is uniformly weak ($R^2 \le 0.034$), despite the large trade-time $\rho_1$: aggregating into fixed time buckets destroys most of the sign signal. **Trade-clock concentrates predictability; wall-clock dilutes it.**
6. **Monotonic, roughly linear price impact** in signed size over the body of the distribution with concave tails on the largest prints (square-root-law regime at the top).
7. **No meaningful cross-symbol lead-lag** in signed flow on this sample: all three pairs peak at lag 0 at 100 ms, and a 10 ms follow-up probe finds only a single 10 ms BTC→SOL lead at peak correlation 0.027.
8. **Realized volatility tracks trade count, not wall-clock time**, confirming Clark-1973 subordinated-time on this tape.
9. **Kyle's $\lambda$:** square-root impact model ($\Delta p = \lambda \sqrt{|q|}$) fits well for BTC ($R^2 = 2.2\%$) and ETH ($R^2 = 1.6\%$); near-zero for SOL.
10. **Bootstrap CIs on $\gamma$** are tight and non-overlapping: BTC [0.78, 0.83], ETH [0.98, 1.04], SOL [0.18, 0.21]. The three regimes are statistically unambiguous.
11. **Intraday seasonality** — clear Asia/London/NY session structure in trade intensity and volatility, peak during London–NY overlap.
12. **Hasbrouck information share** — ETH dominates price discovery over BTC at 5–30s scales. Counterintuitive but robust: the ETH perpetual was the more active price-discovery venue during this 7-day sample.

### Thread E — Live paper-trading validation (NB 09)

The composite signal from Threads A/B is run live via `src/paper_trade.py`, subscribing to Binance websocket streams (`btcusdt@aggTrade` + `btcusdt@bookTicker`), aggregating 1-min bars in real time, and simulating taker fills at live mid ± half-spread with 2 bps round-trip cost.

- ~4 days of data collected (Apr 9–12), 3,657 bars after warmup
- Gross edge: 0.25 bps/turn (backtest was 0.75 — live is lower, consistent with the extended research)
- Gross Sharpe: +4.8 (backtest: +43, but both on short samples with wide CI)
- Net @ 0.5 bps: Sharpe −0.1 (effectively zero)
- Net @ 2 bps: deeply negative, as predicted
- Mean observed half-spread: 0.0085 bps
- **Purpose: pipeline sanity check and confirmation that the backtest was not overfit, not a claim of live profitability.**

### Thread F — Signal stacking (NB 10)

Demonstrates the Grinold–Kahn $\sqrt{N}$ Sharpe scaling with three genuinely uncorrelated signal families:

- **A:** Composite mean-reversion (8 sub-signals from Thread A)
- **B:** Funding-rate carry (4.2 bps edge, only individually profitable signal at 0.5 bps)
- **C:** PCA residual (Thread B)

Pairwise correlations: A–B = 0.001, A–C = 0.25, B–C = 0.004. Equal-weight stack achieves 1.71× Sharpe multiplier (theoretical $\sqrt{3}$ = 1.73×). Extrapolation: a portfolio of 15–20 decorrelated sub-bp signals with $\bar{\rho} = 0.08$ could reach aggregate Sharpe ~6.3.

## Key Findings Across Threads

**The residual mean-reversion signal is real, robust, and structurally sub-bp.**
It survives four methodological variants (1-min time bars, $5M single-asset dollar bars, cross-asset 1-min, cross-asset dollar bars) with lag-1 residual autocorrelation consistently in the $[-0.11, -0.13]$ range. The per-turn edge is ceilinged at roughly $|\rho_1| \sigma^2 \approx 0.08$–$0.20$ bps at the minute-ish horizon — an order of magnitude below any realistic taker cost. The signal belongs as *one component* of a stacked book, not as a standalone strategy.

**Tick-level signed flow is strongly predictable in trade-time.** The classical Lo–MacKinlay power-law decay is present on crypto perps with $\gamma \in [0.78, 0.99]$ for BTC/ETH, directly inside the equity-literature range. This is the single most important tick-level pattern for any maker strategy on the instrument.

**The edge closes only through execution, not through more signal engineering.**
Closing the gap from 0.1 bps to profitable requires (i) maker-side execution and rebate tiers, (ii) queue-position modeling and a realistic fill simulator, or (iii) sub-second horizons where the trade-time sign ACF (not the wall-clock one) is the directly tradable object. All three are infrastructure decisions, not research ones.

## How to Run

```bash
# Create env from environment.yml (pandas, numpy, matplotlib, scipy, statsmodels,
# pyarrow, websockets, jupyter)
conda env create -f environment.yml
conda activate btcusdt-perp-signals

# Pull 12 months of klines, funding, 7 days aggTrades, 30 days bookDepth
python -m src.download_binance

# (Optional) start a live L1 quote capture
python -m src.capture_book_ticker --symbols BTCUSDT ETHUSDT SOLUSDT --hours 48

# Launch notebooks
jupyter notebook notebooks/
```

Notebook order: 01 → 02 → 03 for the original study; 05 → 06 → 07 → 08 for the
extended research threads; 09 for live validation; 10 for signal stacking.
Thread D (NB 08) is the most audience-ready standalone artifact.

## Limitations and Open Questions

- **All backtests are taker-side, no queue modeling.** No fill-probability model, no partial fills, no adverse selection correction. Any maker-side PnL estimate from the current harness would be unreliable.
- **Signed flow is inferred from `is_buyer_maker`**, which is the true aggressor side for `aggTrades` on Binance futures. No reconstruction of book state is attempted.
- **Cross-exchange basis** — the canonical mid-frequency crypto edge — is not yet tested. Feasible with current data.
- **Paper-trading gap:** the websocket harness experienced a ~16-hour outage on Apr 12–13 (Binance websocket connectivity issue). Data before and after the gap is intact; NB 09 handles the gap transparently.
