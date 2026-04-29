"""Phase 6 entry point: statistical and risk analysis of the strategies.

Run from the repository root:

    python experiments/run_risk_analysis.py

Outputs:
    outputs/tables/risk_metrics.csv
    outputs/tables/sharpe_confidence_intervals.csv
    outputs/figures/bootstrap_nav_paths.png
    outputs/logs/risk_analysis_log.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.backtester import Backtester  # noqa: E402
from src.config import DATA_PATH, FIGURE_DIR, LOG_DIR, TABLE_DIR  # noqa: E402
from src.data_loader import compute_returns, load_price_data  # noqa: E402
from src.metrics import (  # noqa: E402
    annualized_volatility,
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
)
from src.risk import (  # noqa: E402
    bootstrap_nav_paths,
    conditional_value_at_risk,
    estimate_tail_loss_probability,
    importance_sampling_tail_stress,
    sharpe_confidence_interval,
    value_at_risk,
)
from src.strategies.bandit import UCBMetaStrategy  # noqa: E402
from src.strategies.benchmarks import (  # noqa: E402
    SMACrossoverPortfolioStrategy,
    TopKMomentumStrategy,
)
from src.strategies.cross_sectional import (  # noqa: E402
    LowVolatilityMomentumStrategy,
    RiskAdjustedMomentumStrategy,
)


def _build_arms() -> list:
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


def _strategy_zoo() -> list:
    arms = _build_arms()
    ucb = UCBMetaStrategy(arms=_build_arms(), c=1.0)
    return arms + [ucb]


def main() -> None:
    print(f"[phase6] Loading price data from {DATA_PATH}")
    prices = load_price_data(DATA_PATH)
    returns = compute_returns(prices)
    print(
        f"[phase6] Loaded {prices.shape[1]} stocks across "
        f"{len(prices)} trading days "
        f"({prices.index.min().date()} -> {prices.index.max().date()})"
    )

    bt = Backtester(prices, returns=returns)
    strategies = _strategy_zoo()

    risk_rows = []
    sharpe_ci_rows = []
    daily_returns_by_strategy = {}

    for strat in strategies:
        print(f"[phase6] Running {strat.name}")
        result = bt.run(strat)
        r = result.daily_returns
        daily_returns_by_strategy[strat.name] = (result, r)

        var5 = value_at_risk(r, alpha=0.05)
        cvar5 = conditional_value_at_risk(r, alpha=0.05)
        tail_p = estimate_tail_loss_probability(r, loss_threshold=-0.05)
        stress = importance_sampling_tail_stress(
            r, loss_threshold=-0.05, stress_multiplier=3.0, seed=42
        )
        ci = sharpe_confidence_interval(r, confidence=0.95)

        risk_rows.append(
            {
                "strategy": strat.name,
                "cumulative_return": cumulative_return(r),
                "annualized_volatility": annualized_volatility(r),
                "sharpe_ratio": sharpe_ratio(r),
                "max_drawdown": max_drawdown(result.nav),
                "var_5pct": var5,
                "cvar_5pct": cvar5,
                "p_daily_loss_below_5pct": tail_p,
                "stressed_tail_probability": stress["stressed_tail_probability"],
                "stressed_cvar": stress["stressed_cvar"],
                "sharpe_ci_lower": ci["lower_ci"],
                "sharpe_ci_upper": ci["upper_ci"],
            }
        )
        sharpe_ci_rows.append(
            {
                "strategy": strat.name,
                "sharpe": ci["sharpe"],
                "standard_error": ci["standard_error"],
                "lower_ci": ci["lower_ci"],
                "upper_ci": ci["upper_ci"],
            }
        )

    risk_df = pd.DataFrame(risk_rows).sort_values("sharpe_ratio", ascending=False)
    risk_path = TABLE_DIR / "risk_metrics.csv"
    risk_df.to_csv(risk_path, index=False)

    ci_df = pd.DataFrame(sharpe_ci_rows).sort_values("sharpe", ascending=False)
    ci_path = TABLE_DIR / "sharpe_confidence_intervals.csv"
    ci_df.to_csv(ci_path, index=False)

    # Bootstrap NAV paths for the best-Sharpe strategy.
    best_name = str(risk_df.iloc[0]["strategy"])
    _result, best_returns = daily_returns_by_strategy[best_name]
    print(f"[phase6] Bootstrapping NAV paths for best-Sharpe strategy: {best_name}")
    nav_paths = bootstrap_nav_paths(
        best_returns, n_paths=1000, horizon=len(best_returns), seed=42
    )

    boot_path = FIGURE_DIR / "bootstrap_nav_paths.png"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    rng = np.random.default_rng(0)
    if nav_paths.shape[1] >= 1:
        n_show = min(50, nav_paths.shape[1])
        cols = list(rng.choice(nav_paths.columns, size=n_show, replace=False))
        for c in cols:
            ax.plot(nav_paths.index, nav_paths[c].values, color="tab:blue",
                    alpha=0.15, linewidth=0.7)
        median_path = nav_paths.median(axis=1)
        ax.plot(median_path.index, median_path.values, color="black",
                linewidth=2.0, label="Median path")
        ax.legend(loc="best", fontsize=9, frameon=False)
    ax.set_title(f"Bootstrap NAV paths — {best_name} (50 of {nav_paths.shape[1]})")
    ax.set_xlabel("Trading-day index")
    ax.set_ylabel("NAV (initial = 1.0)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(boot_path, dpi=150)
    plt.close(fig)

    log_payload = {
        "n_days": int(len(prices)),
        "n_stocks": int(prices.shape[1]),
        "start": str(prices.index.min().date()),
        "end": str(prices.index.max().date()),
        "best_sharpe_strategy": best_name,
        "risk_metrics": risk_df.to_dict(orient="records"),
        "sharpe_confidence_intervals": ci_df.to_dict(orient="records"),
        "bootstrap": {
            "n_paths": int(nav_paths.shape[1]),
            "horizon": int(nav_paths.shape[0] - 1),
            "median_terminal_nav": (
                float(nav_paths.iloc[-1].median()) if not nav_paths.empty else None
            ),
            "p05_terminal_nav": (
                float(np.quantile(nav_paths.iloc[-1].values, 0.05))
                if not nav_paths.empty
                else None
            ),
            "p95_terminal_nav": (
                float(np.quantile(nav_paths.iloc[-1].values, 0.95))
                if not nav_paths.empty
                else None
            ),
        },
    }
    log_path = LOG_DIR / "risk_analysis_log.json"
    with open(log_path, "w") as f:
        json.dump(log_payload, f, indent=2)

    print("\n=== Risk metrics (sorted by Sharpe) ===")
    with pd.option_context(
        "display.max_columns", None, "display.width", 240,
        "display.float_format", "{:.4f}".format,
    ):
        print(risk_df.to_string(index=False))

    print("\n=== Sharpe 95% confidence intervals ===")
    with pd.option_context(
        "display.max_columns", None, "display.width", 200,
        "display.float_format", "{:.4f}".format,
    ):
        print(ci_df.to_string(index=False))

    print(f"\n[phase6] Wrote: {risk_path}")
    print(f"[phase6] Wrote: {ci_path}")
    print(f"[phase6] Wrote: {boot_path}")
    print(f"[phase6] Wrote: {log_path}")


if __name__ == "__main__":
    main()
