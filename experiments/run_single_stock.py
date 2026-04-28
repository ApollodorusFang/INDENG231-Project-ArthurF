"""Phase 2 entry point: compare five single-stock strategies on one ticker.

Run from the repository root:

    python experiments/run_single_stock.py

Outputs:
    outputs/tables/single_stock_metrics.csv
    outputs/figures/single_stock_nav.png
    outputs/figures/single_stock_drawdown.png
    outputs/logs/single_stock_log.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.backtester import Backtester  # noqa: E402
from src.config import DATA_PATH, FIGURE_DIR, LOG_DIR, TABLE_DIR  # noqa: E402
from src.data_loader import compute_returns, load_price_data  # noqa: E402
from src.metrics import summarize_performance  # noqa: E402
from src.plotting import plot_multiple_drawdowns, plot_multiple_nav  # noqa: E402
from src.strategies.single_stock import (  # noqa: E402
    MovingAverageCrossoverStrategy,
    SingleStockMeanReversionStrategy,
    SingleStockMomentumStrategy,
    VolatilityFilteredMomentumStrategy,
    ZScoreMeanReversionStrategy,
)

PREFERRED_TICKERS = ("NVDA", "AAPL")


def pick_ticker(prices: pd.DataFrame) -> str:
    """Choose NVDA -> AAPL -> first available column, in that order."""
    for t in PREFERRED_TICKERS:
        if t in prices.columns:
            return t
    return str(prices.columns[0])


def build_strategies(ticker: str) -> list:
    return [
        SingleStockMomentumStrategy(ticker, lookback=20),
        SingleStockMeanReversionStrategy(ticker, lookback=20, threshold=-0.05),
        MovingAverageCrossoverStrategy(ticker, short_window=20, long_window=50),
        VolatilityFilteredMomentumStrategy(
            ticker,
            lookback=20,
            volatility_window=20,
            volatility_threshold=0.03,
        ),
        ZScoreMeanReversionStrategy(ticker, window=20, z_threshold=-1.0),
    ]


def main() -> None:
    print(f"[phase2] Loading price data from {DATA_PATH}")
    prices = load_price_data(DATA_PATH)
    returns = compute_returns(prices)
    print(
        f"[phase2] Loaded {prices.shape[1]} stocks across "
        f"{len(prices)} trading days "
        f"({prices.index.min().date()} -> {prices.index.max().date()})"
    )

    ticker = pick_ticker(prices)
    print(f"[phase2] Selected ticker: {ticker}")

    bt = Backtester(prices, returns=returns)
    strategies = build_strategies(ticker)

    results = {}
    rows = []
    for strat in strategies:
        print(f"[phase2] Running {strat.name}")
        result = bt.run(strat)
        results[strat.name] = result
        rows.append(summarize_performance(result))

    metrics_df = pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False)
    metrics_path = TABLE_DIR / "single_stock_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    nav_path = FIGURE_DIR / "single_stock_nav.png"
    dd_path = FIGURE_DIR / "single_stock_drawdown.png"
    plot_multiple_nav(
        results, nav_path, title=f"Single-stock strategies — {ticker}"
    )
    plot_multiple_drawdowns(
        results, dd_path, title=f"Single-stock drawdowns — {ticker}"
    )

    log_path = LOG_DIR / "single_stock_log.json"
    log_payload = {
        "ticker": ticker,
        "n_days": int(len(prices)),
        "start": str(prices.index.min().date()),
        "end": str(prices.index.max().date()),
        "metrics": metrics_df.to_dict(orient="records"),
    }
    with open(log_path, "w") as f:
        json.dump(log_payload, f, indent=2)

    print("\n=== Single-stock metrics (sorted by Sharpe) ===")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.float_format",
        "{:.4f}".format,
    ):
        print(metrics_df.to_string(index=False))

    print(f"\n[phase2] Wrote: {metrics_path}")
    print(f"[phase2] Wrote: {nav_path}")
    print(f"[phase2] Wrote: {dd_path}")
    print(f"[phase2] Wrote: {log_path}")


if __name__ == "__main__":
    main()
