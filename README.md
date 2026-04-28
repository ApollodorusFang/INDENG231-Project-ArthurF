# INDENG231-Project-ArthurF

A modular daily-close backtesting simulation system for Nasdaq-100
constituent stocks, built for INDENG 231 Course Project 1.

## Overview

The system is designed around three pieces:

* **Data layer** ([src/data_loader.py](src/data_loader.py)) — loads a
  wide or long CSV of daily close prices and produces a clean,
  date-indexed price panel plus simple-return panel.
* **Strategy layer**
  ([src/strategies/base.py](src/strategies/base.py)) — a
  `BaseStrategy` that maps the information available at the close of
  day *t* (prices and returns up to and including *t*) to a target
  weight vector. Strategies override `generate_weights`; the base
  class enforces the no-short / no-leverage constraints in
  `validate_weights`.
* **Engine layer** ([src/backtester.py](src/backtester.py)) — iterates
  trading days, calls the strategy with only the history through *t*,
  and applies the resulting weights to the *next* day's returns. This
  is what keeps the system free of look-ahead bias.

Outputs (NAV curves, drawdown plots, metric tables, run logs) land
under [outputs/](outputs/) so experiments are easy to compare.

## Hard constraints

1. Daily-close decisions and execution only.
2. At date *t*, only data up to and including *t* is visible to the
   strategy.
3. Target weights chosen at *t* are realized against returns from
   *t* to *t+1*.
4. No short selling: weights ≥ 0.
5. No leverage: sum of weights ≤ 1; the residual is cash.
6. If no stock is selected, the portfolio holds 100% cash.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `requirements.txt` for Phase 1 only needs:

```
pandas
numpy
matplotlib
```

## Data placement

Place the dataset at:

```
data/nasdaq100_daily_5y.csv
```

The loader supports both layouts:

* **Wide** — first column is the date, every other column is a ticker
  with close prices.
* **Long** — one row per `(date, ticker)` pair with a price column
  named `close`, `adj_close`, or `price`.

Tickers are read from the file; nothing is hard-coded.

## Phase 1 — run command

From the repository root:

```bash
python experiments/run_all.py
```

This will:

1. Load the dataset.
2. Run the `EqualWeightBuyAndHoldStrategy` smoke test over every
   stock available in the file.
3. Write:

   * [outputs/tables/phase1_metrics.csv](outputs/tables/) — single-row
     metrics table.
   * [outputs/figures/phase1_nav.png](outputs/figures/) — NAV curve.
   * [outputs/figures/phase1_drawdown.png](outputs/figures/) —
     drawdown curve.
   * [outputs/logs/phase1_log.json](outputs/logs/) — run summary.

## Phase 2 — single-stock strategy comparison

Phase 2 picks one stock (default: `NVDA`, falling back to `AAPL`, then
the first available ticker) and runs five long-or-cash strategies on
it through the same backtester:

* **Momentum** — long if the trailing 20-day return is positive.
* **Mean reversion** — long if the trailing 20-day return is below a
  threshold (default −5%).
* **MA crossover** — long if the 20-day SMA is above the 50-day SMA.
* **Volatility-filtered momentum** — long only when momentum is
  positive *and* trailing 20-day volatility is below 3%.
* **Z-score mean reversion** — long when the 20-day price z-score
  drops below −1 standard deviation.

Run from the repository root:

```bash
python experiments/run_single_stock.py
```

This produces:

* [outputs/tables/single_stock_metrics.csv](outputs/tables/) — one row
  per strategy, sorted by Sharpe.
* [outputs/figures/single_stock_nav.png](outputs/figures/) — overlaid
  NAV curves.
* [outputs/figures/single_stock_drawdown.png](outputs/figures/) —
  overlaid drawdown curves.
* [outputs/logs/single_stock_log.json](outputs/logs/) — run summary.

## Phase 3 — portfolio benchmark strategies

Phase 3 implements three cross-sectional portfolio strategies over the
full Nasdaq-100 universe, plus the underlying portfolio-construction
helpers in [src/portfolio.py](src/portfolio.py)
(`equal_weight`, `inverse_volatility_weight`, `rank_weight`).

* **SMA crossover (portfolio)** — for each stock, compare the 20-day
  and 50-day SMA; equal-weight the stocks where the short SMA is above
  the long SMA. Cash if none qualify.
* **Top-K momentum** — rank every stock by trailing 30-day return,
  equal-weight the top 10. (Project benchmark #2.)
* **Risk-adjusted top-K momentum** — rank by
  `trailing_return / rolling_volatility`, keep the top 10 with a
  positive signal, weight them inverse-vol.

Run from the repository root:

```bash
python experiments/run_benchmarks.py
```

This produces:

* [outputs/tables/benchmark_metrics.csv](outputs/tables/) — one row per
  strategy, sorted by Sharpe.
* [outputs/figures/benchmark_nav.png](outputs/figures/) — overlaid NAV
  curves.
* [outputs/figures/benchmark_drawdown.png](outputs/figures/) — overlaid
  drawdown curves.
* [outputs/logs/benchmark_log.json](outputs/logs/) — run summary.

## Phase 4 — new strategies vs. benchmarks

Phase 4 introduces two new portfolio strategies designed to beat the
Phase 3 benchmark Sharpe ratios, and compares them on a held-out test
window:

* **RiskAdjustedMomentumStrategy** — top-K by `return / volatility`
  signal, weighted inverse-vol, with an optional market-volatility
  throttle that scales gross exposure to `defensive_exposure` when
  realized market vol exceeds `market_vol_threshold`.
* **LowVolatilityMomentumStrategy** — combines trailing-return rank
  (higher is better) with low-volatility rank (lower vol is better)
  into a transparent additive score, picks top-K, weights inverse-vol.

The script does a small grid search
(`lookback ∈ {30, 60, 90}`, `volatility_window ∈ {20, 40}`,
`k ∈ {5, 10, 15}`) on the first 60% of dates, then evaluates the tuned
strategies — alongside `SMACrossoverPortfolioStrategy` and
`TopKMomentumStrategy` — on the remaining 40%.

Run from the repository root:

```bash
python experiments/run_new_strategies.py
```

This produces:

* [outputs/tables/benchmarks_vs_new_metrics.csv](outputs/tables/) —
  test-window metrics, sorted by Sharpe.
* [outputs/figures/benchmarks_vs_new_nav.png](outputs/figures/) —
  overlaid NAV curves.
* [outputs/figures/benchmarks_vs_new_drawdown.png](outputs/figures/) —
  overlaid drawdown curves.
* [outputs/logs/new_strategies_log.json](outputs/logs/) — run summary
  including tuned parameters and the beats-benchmarks verdict.

## Repository layout

```
src/
  config.py            # paths and constants
  data_loader.py       # CSV ingest + return computation
  backtester.py        # Backtester + BacktestResult
  metrics.py           # cumulative/annualized return, Sharpe, drawdown, …
  plotting.py          # NAV and drawdown plots
  portfolio.py         # equal / inverse-vol / rank weighting helpers
  strategies/
    base.py            # BaseStrategy + EqualWeightBuyAndHoldStrategy
    single_stock.py    # Phase 2 single-stock long-or-cash strategies
    benchmarks.py      # Phase 3 portfolio benchmark strategies
    cross_sectional.py # Phase 4 risk-adjusted / low-vol momentum
experiments/
  run_all.py           # Phase 1 entry point
  run_single_stock.py  # Phase 2 entry point
  run_benchmarks.py    # Phase 3 entry point
  run_new_strategies.py # Phase 4 entry point
data/                  # input CSV (gitignored)
outputs/               # generated artifacts (gitignored)
```
