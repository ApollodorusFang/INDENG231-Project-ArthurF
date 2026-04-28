"""Phase 4 entry point: two new strategies vs. the Phase 3 benchmarks.

Run from the repository root:

    python experiments/run_new_strategies.py

Outputs:
    outputs/tables/benchmarks_vs_new_metrics.csv
    outputs/figures/benchmarks_vs_new_nav.png
    outputs/figures/benchmarks_vs_new_drawdown.png
    outputs/logs/new_strategies_log.json

The script also prints whether each new strategy beats *both*
benchmark Sharpe ratios on the test (out-of-sample) window.
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.backtester import Backtester  # noqa: E402
from src.config import DATA_PATH, FIGURE_DIR, LOG_DIR, TABLE_DIR  # noqa: E402
from src.data_loader import compute_returns, load_price_data  # noqa: E402
from src.metrics import sharpe_ratio, summarize_performance  # noqa: E402
from src.plotting import plot_multiple_drawdowns, plot_multiple_nav  # noqa: E402
from src.strategies.benchmarks import (  # noqa: E402
    SMACrossoverPortfolioStrategy,
    TopKMomentumStrategy,
)
from src.strategies.cross_sectional import (  # noqa: E402
    LowVolatilityMomentumStrategy,
    RiskAdjustedMomentumStrategy,
)

# Conservative grids -- kept small so the search stays transparent.
LOOKBACK_GRID = (30, 60, 90)
VOL_WINDOW_GRID = (20, 40)
K_GRID = (5, 10, 15)

# 60% in-sample / 40% out-of-sample split.
TRAIN_FRACTION = 0.60


def _split_train_test(prices: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    n = len(prices)
    train_end_idx = max(1, int(n * TRAIN_FRACTION) - 1)
    return (
        prices.index[0],
        prices.index[train_end_idx],
        prices.index[-1],
    )


def _grid_search_sharpe(
    factory,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    train_end: pd.Timestamp,
    label: str,
):
    """Brute-force the small grid on the in-sample window; return best params."""
    bt_train = Backtester(prices, returns=returns, end=train_end)
    best = None
    for lb, vw, k in product(LOOKBACK_GRID, VOL_WINDOW_GRID, K_GRID):
        try:
            strat = factory(lb, vw, k)
            res = bt_train.run(strat)
            sr = sharpe_ratio(res.daily_returns)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label}] lb={lb} vw={vw} k={k} -> error: {exc!r}")
            continue
        if best is None or sr > best[0]:
            best = (sr, (lb, vw, k))
    if best is None:
        raise RuntimeError(f"No parameter set worked for {label}")
    sr, params = best
    print(
        f"  [{label}] best in-sample Sharpe={sr:.4f} "
        f"with lookback={params[0]}, volatility_window={params[1]}, k={params[2]}"
    )
    return params


def _build_risk_adjusted(lb: int, vw: int, k: int) -> RiskAdjustedMomentumStrategy:
    return RiskAdjustedMomentumStrategy(
        lookback=lb,
        volatility_window=vw,
        k=k,
        market_vol_window=20,
        market_vol_threshold=0.02,
        defensive_exposure=0.5,
    )


def _build_low_vol_momentum(lb: int, vw: int, k: int) -> LowVolatilityMomentumStrategy:
    return LowVolatilityMomentumStrategy(lookback=lb, volatility_window=vw, k=k)


def main() -> None:
    print(f"[phase4] Loading price data from {DATA_PATH}")
    prices = load_price_data(DATA_PATH)
    returns = compute_returns(prices)
    print(
        f"[phase4] Loaded {prices.shape[1]} stocks across "
        f"{len(prices)} trading days "
        f"({prices.index.min().date()} -> {prices.index.max().date()})"
    )

    start, train_end, _end = _split_train_test(prices)
    test_start_idx = prices.index.get_loc(train_end) + 1
    test_start = prices.index[min(test_start_idx, len(prices) - 1)]
    print(
        f"[phase4] Train window: {start.date()} -> {train_end.date()} "
        f"({test_start_idx} days)"
    )
    print(
        f"[phase4] Test  window: {test_start.date()} -> {prices.index[-1].date()} "
        f"({len(prices) - test_start_idx} days)"
    )

    print("\n[phase4] Grid search on the train window:")
    risk_adj_params = _grid_search_sharpe(
        _build_risk_adjusted, prices, returns, train_end, "RiskAdjustedMomentum"
    )
    low_vol_params = _grid_search_sharpe(
        _build_low_vol_momentum, prices, returns, train_end, "LowVolatilityMomentum"
    )

    # Final test-window evaluation: benchmarks + tuned new strategies.
    bt_test = Backtester(prices, returns=returns, start=test_start)
    strategies = [
        SMACrossoverPortfolioStrategy(short_window=20, long_window=50),
        TopKMomentumStrategy(lookback=30, k=10),
        _build_risk_adjusted(*risk_adj_params),
        _build_low_vol_momentum(*low_vol_params),
    ]

    print("\n[phase4] Running on test window:")
    results = {}
    rows = []
    for strat in strategies:
        print(f"  - {strat.name}")
        result = bt_test.run(strat)
        results[strat.name] = result
        rows.append(summarize_performance(result))

    metrics_df = pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False)
    metrics_path = TABLE_DIR / "benchmarks_vs_new_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    nav_path = FIGURE_DIR / "benchmarks_vs_new_nav.png"
    dd_path = FIGURE_DIR / "benchmarks_vs_new_drawdown.png"
    plot_multiple_nav(
        results, nav_path, title="Phase 4: benchmarks vs. new strategies"
    )
    plot_multiple_drawdowns(
        results, dd_path, title="Phase 4: drawdown comparison"
    )

    # Compare new strategies to the two benchmarks by Sharpe ratio.
    sharpe_lookup = dict(zip(metrics_df["strategy"], metrics_df["sharpe_ratio"]))
    bench_sharpes = [
        sharpe_lookup[s.name]
        for s in strategies
        if isinstance(s, (SMACrossoverPortfolioStrategy, TopKMomentumStrategy))
    ]
    bench_max = max(bench_sharpes)

    new_strategies = [s for s in strategies if isinstance(
        s, (RiskAdjustedMomentumStrategy, LowVolatilityMomentumStrategy)
    )]
    comparison = []
    for s in new_strategies:
        sr = sharpe_lookup[s.name]
        beats = sr > bench_max
        comparison.append({"strategy": s.name, "sharpe": sr, "beats_benchmarks": beats})

    log_path = LOG_DIR / "new_strategies_log.json"
    log_payload = {
        "n_days": int(len(prices)),
        "n_stocks": int(prices.shape[1]),
        "train_window": {"start": str(start.date()), "end": str(train_end.date())},
        "test_window": {"start": str(test_start.date()), "end": str(prices.index[-1].date())},
        "tuned_params": {
            "RiskAdjustedMomentumStrategy": {
                "lookback": risk_adj_params[0],
                "volatility_window": risk_adj_params[1],
                "k": risk_adj_params[2],
            },
            "LowVolatilityMomentumStrategy": {
                "lookback": low_vol_params[0],
                "volatility_window": low_vol_params[1],
                "k": low_vol_params[2],
            },
        },
        "metrics": metrics_df.to_dict(orient="records"),
        "comparison_vs_benchmarks": comparison,
    }
    with open(log_path, "w") as f:
        json.dump(log_payload, f, indent=2)

    print("\n=== Test-window metrics (sorted by Sharpe) ===")
    with pd.option_context(
        "display.max_columns", None, "display.width", 220,
        "display.float_format", "{:.4f}".format,
    ):
        print(metrics_df.to_string(index=False))

    print("\n=== New strategies vs. benchmarks (Sharpe) ===")
    print(f"  Benchmark max Sharpe = {bench_max:.4f}")
    for row in comparison:
        verdict = "BEATS" if row["beats_benchmarks"] else "does NOT beat"
        print(f"  {row['strategy']}: Sharpe={row['sharpe']:.4f} -> {verdict} both benchmarks")

    print(f"\n[phase4] Wrote: {metrics_path}")
    print(f"[phase4] Wrote: {nav_path}")
    print(f"[phase4] Wrote: {dd_path}")
    print(f"[phase4] Wrote: {log_path}")


if __name__ == "__main__":
    main()
