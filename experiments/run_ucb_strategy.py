"""Phase 5 entry point: UCB meta-strategy over four base strategies.

Run from the repository root:

    python experiments/run_ucb_strategy.py

Outputs:
    outputs/tables/ucb_metrics.csv
    outputs/figures/ucb_nav.png
    outputs/figures/ucb_drawdown.png
    outputs/logs/ucb_log.json

The script also prints arm-selection counts and mean realized rewards
for the UCB meta-strategy.
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
from src.strategies.bandit import UCBMetaStrategy  # noqa: E402
from src.strategies.benchmarks import (  # noqa: E402
    SMACrossoverPortfolioStrategy,
    TopKMomentumStrategy,
)
from src.strategies.cross_sectional import (  # noqa: E402
    LowVolatilityMomentumStrategy,
    RiskAdjustedMomentumStrategy,
)


def build_arms() -> list:
    """Independent arm instances -- the UCB strategy mutates none of them."""
    return [
        SMACrossoverPortfolioStrategy(short_window=20, long_window=50),
        TopKMomentumStrategy(lookback=30, k=10),
        RiskAdjustedMomentumStrategy(
            lookback=60,
            volatility_window=20,
            k=10,
            market_vol_window=20,
            market_vol_threshold=0.02,
            defensive_exposure=0.5,
        ),
        LowVolatilityMomentumStrategy(lookback=60, volatility_window=20, k=10),
    ]


def main() -> None:
    print(f"[phase5] Loading price data from {DATA_PATH}")
    prices = load_price_data(DATA_PATH)
    returns = compute_returns(prices)
    print(
        f"[phase5] Loaded {prices.shape[1]} stocks across "
        f"{len(prices)} trading days "
        f"({prices.index.min().date()} -> {prices.index.max().date()})"
    )

    bt = Backtester(prices, returns=returns)

    # Standalone runs of each base strategy (fresh instances).
    base_strategies = build_arms()
    # Independent arm instances for the meta-strategy.
    ucb = UCBMetaStrategy(arms=build_arms(), c=1.0)

    all_strategies = list(base_strategies) + [ucb]

    results = {}
    rows = []
    for strat in all_strategies:
        print(f"[phase5] Running {strat.name}")
        result = bt.run(strat)
        results[strat.name] = result
        rows.append(summarize_performance(result))

    metrics_df = pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False)
    metrics_path = TABLE_DIR / "ucb_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    nav_path = FIGURE_DIR / "ucb_nav.png"
    dd_path = FIGURE_DIR / "ucb_drawdown.png"
    plot_multiple_nav(results, nav_path, title="Phase 5: UCB meta-strategy vs. arms")
    plot_multiple_drawdowns(results, dd_path, title="Phase 5: drawdown comparison")

    selection_counts = ucb.selection_counts()
    mean_rewards = ucb.mean_rewards()

    log_path = LOG_DIR / "ucb_log.json"
    log_payload = {
        "n_days": int(len(prices)),
        "n_stocks": int(prices.shape[1]),
        "start": str(prices.index.min().date()),
        "end": str(prices.index.max().date()),
        "ucb_c": ucb.c,
        "metrics": metrics_df.to_dict(orient="records"),
        "ucb_selection_counts": selection_counts,
        "ucb_mean_rewards": mean_rewards,
        "ucb_total_trials": ucb.total_trials,
    }
    with open(log_path, "w") as f:
        json.dump(log_payload, f, indent=2)

    print("\n=== Metrics (sorted by Sharpe) ===")
    with pd.option_context(
        "display.max_columns", None, "display.width", 220,
        "display.float_format", "{:.4f}".format,
    ):
        print(metrics_df.to_string(index=False))

    print("\n=== UCB arm selection counts ===")
    total = max(ucb.total_trials, 1)
    for name, n in selection_counts.items():
        share = n / total
        mean_r = mean_rewards.get(name, 0.0)
        print(
            f"  {name:>50}: {n:>5d} selections "
            f"({share:6.1%})  mean_reward={mean_r:+.6f}"
        )
    print(f"  total trials: {ucb.total_trials}")

    print(f"\n[phase5] Wrote: {metrics_path}")
    print(f"[phase5] Wrote: {nav_path}")
    print(f"[phase5] Wrote: {dd_path}")
    print(f"[phase5] Wrote: {log_path}")


if __name__ == "__main__":
    main()
