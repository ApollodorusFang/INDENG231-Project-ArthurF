"""Phase 3 entry point: run the three portfolio benchmark strategies.

Run from the repository root:

    python experiments/run_benchmarks.py

Outputs:
    outputs/tables/benchmark_metrics.csv
    outputs/figures/benchmark_nav.png
    outputs/figures/benchmark_drawdown.png
    outputs/logs/benchmark_log.json
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
from src.strategies.benchmarks import (  # noqa: E402
    RiskAdjustedTopKMomentumStrategy,
    SMACrossoverPortfolioStrategy,
    TopKMomentumStrategy,
)


def build_strategies() -> list:
    return [
        SMACrossoverPortfolioStrategy(short_window=20, long_window=50),
        TopKMomentumStrategy(lookback=30, k=10),
        RiskAdjustedTopKMomentumStrategy(lookback=30, volatility_window=20, k=10),
    ]


def main() -> None:
    print(f"[phase3] Loading price data from {DATA_PATH}")
    prices = load_price_data(DATA_PATH)
    returns = compute_returns(prices)
    print(
        f"[phase3] Loaded {prices.shape[1]} stocks across "
        f"{len(prices)} trading days "
        f"({prices.index.min().date()} -> {prices.index.max().date()})"
    )

    bt = Backtester(prices, returns=returns)
    strategies = build_strategies()

    results = {}
    rows = []
    for strat in strategies:
        print(f"[phase3] Running {strat.name}")
        result = bt.run(strat)
        results[strat.name] = result
        rows.append(summarize_performance(result))

    metrics_df = pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False)
    metrics_path = TABLE_DIR / "benchmark_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    nav_path = FIGURE_DIR / "benchmark_nav.png"
    dd_path = FIGURE_DIR / "benchmark_drawdown.png"
    plot_multiple_nav(results, nav_path, title="Benchmark portfolio strategies")
    plot_multiple_drawdowns(results, dd_path, title="Benchmark drawdowns")

    log_path = LOG_DIR / "benchmark_log.json"
    log_payload = {
        "n_days": int(len(prices)),
        "n_stocks": int(prices.shape[1]),
        "start": str(prices.index.min().date()),
        "end": str(prices.index.max().date()),
        "metrics": metrics_df.to_dict(orient="records"),
    }
    with open(log_path, "w") as f:
        json.dump(log_payload, f, indent=2)

    print("\n=== Benchmark metrics (sorted by Sharpe) ===")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.float_format",
        "{:.4f}".format,
    ):
        print(metrics_df.to_string(index=False))

    print(f"\n[phase3] Wrote: {metrics_path}")
    print(f"[phase3] Wrote: {nav_path}")
    print(f"[phase3] Wrote: {dd_path}")
    print(f"[phase3] Wrote: {log_path}")


if __name__ == "__main__":
    main()
