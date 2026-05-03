"""Phase 4 entry point: two new strategies vs. the project benchmarks.

Run from the repository root:

    python experiments/run_new_strategies.py

Outputs:
    outputs/tables/benchmarks_vs_new_metrics.csv
    outputs/figures/benchmarks_vs_new_nav.png
    outputs/figures/benchmarks_vs_new_drawdown.png
    outputs/logs/new_strategies_log.json

Protocol
--------
1. Tune each new strategy on the first 60% of dates (in-sample) with a
   small grid; the parameter triple/tuple maximizing in-sample Sharpe is
   selected.
2. Re-run the tuned strategies on the held-out 40% test window alongside
   the two project benchmarks (SMA crossover, top-K momentum).
3. The verdict (does each new strategy beat *both* benchmarks by Sharpe
   on the held-out test window?) is written to the JSON log.

The two new strategies submitted for Deliverable 5 are:

* ``ConcentratedMomentumStrategy`` -- top-K raw-momentum names, gated to
  positive trailing return, equal- or inverse-vol-weighted.
* ``MomentumWithDrawdownControlStrategy`` -- top-K momentum with an
  equal-weight-market drawdown overlay that scales exposure down when
  the broad market is in a sustained drawdown.
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
    ConcentratedMomentumStrategy,
    MomentumWithDrawdownControlStrategy,
)

# 60% in-sample / 40% out-of-sample split.
TRAIN_FRACTION = 0.60

# Grids (kept transparent and small).
CONCENTRATED_GRID = {
    "lookback": (20, 30, 45, 60, 90),
    "k": (3, 5, 7, 10, 15),
    "weighting": ("equal", "inverse_vol"),
    "volatility_window": (20,),
}
DRAWDOWN_GRID = {
    "lookback": (20, 30, 45, 60, 90),
    "k": (3, 5, 7, 10),
    "drawdown_threshold": (-0.05, -0.10, -0.15),
    "defensive_exposure": (0.5,),
    "market_window": (120,),
}


def _split_train_test(
    prices: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    n = len(prices)
    train_end_idx = max(1, int(n * TRAIN_FRACTION) - 1)
    return (
        prices.index[0],
        prices.index[train_end_idx],
        prices.index[-1],
    )


def _grid_search(
    factory,
    grid: dict,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    train_end: pd.Timestamp,
    label: str,
) -> dict:
    """Brute-force a small grid on the in-sample window; return best params."""
    bt_train = Backtester(prices, returns=returns, end=train_end)
    keys = list(grid.keys())
    best = None
    for values in product(*[grid[k] for k in keys]):
        params = dict(zip(keys, values))
        try:
            strat = factory(**params)
            res = bt_train.run(strat)
            sr = sharpe_ratio(res.daily_returns)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label}] {params} -> error: {exc!r}")
            continue
        if best is None or sr > best[0]:
            best = (sr, params)
    if best is None:
        raise RuntimeError(f"No parameter set worked for {label}")
    sr, params = best
    pretty = ", ".join(f"{k}={v}" for k, v in params.items())
    print(f"  [{label}] best in-sample Sharpe={sr:.4f} with {pretty}")
    return params


def _short_label(strategy) -> str:
    """Compact display label for figures and verdict tables."""
    if isinstance(strategy, SMACrossoverPortfolioStrategy):
        return "B1: SMA crossover"
    if isinstance(strategy, TopKMomentumStrategy):
        return f"B2: Top-{strategy.k} momentum"
    if isinstance(strategy, ConcentratedMomentumStrategy):
        w = "EW" if strategy.weighting == "equal" else "IV"
        return f"NEW1: Concentrated mom (k={strategy.k}, lb={strategy.lookback}, {w})"
    if isinstance(strategy, MomentumWithDrawdownControlStrategy):
        return (
            f"NEW2: Momentum + DD ctrl "
            f"(k={strategy.k}, lb={strategy.lookback}, "
            f"thr={strategy.drawdown_threshold:g})"
        )
    return strategy.name


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
    concentrated_params = _grid_search(
        ConcentratedMomentumStrategy,
        CONCENTRATED_GRID,
        prices,
        returns,
        train_end,
        "ConcentratedMomentum",
    )
    drawdown_params = _grid_search(
        MomentumWithDrawdownControlStrategy,
        DRAWDOWN_GRID,
        prices,
        returns,
        train_end,
        "MomentumWithDrawdownControl",
    )

    # Final test-window evaluation: benchmarks + tuned new strategies.
    bt_test = Backtester(prices, returns=returns, start=test_start)
    benchmarks = [
        SMACrossoverPortfolioStrategy(short_window=20, long_window=50),
        TopKMomentumStrategy(lookback=30, k=10),
    ]
    new_strategies = [
        ConcentratedMomentumStrategy(**concentrated_params),
        MomentumWithDrawdownControlStrategy(**drawdown_params),
    ]
    strategies = benchmarks + new_strategies

    print("\n[phase4] Running on test window:")
    results = {}
    rows = []
    for strat in strategies:
        print(f"  - {strat.name}")
        result = bt_test.run(strat)
        # Use the short label so figure legends are readable.
        results[_short_label(strat)] = result
        rows.append(summarize_performance(result))

    metrics_df = pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False)
    metrics_path = TABLE_DIR / "benchmarks_vs_new_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    nav_path = FIGURE_DIR / "benchmarks_vs_new_nav.png"
    dd_path = FIGURE_DIR / "benchmarks_vs_new_drawdown.png"
    plot_multiple_nav(
        results,
        nav_path,
        title="Phase 4: benchmarks vs. new strategies (test window)",
    )
    plot_multiple_drawdowns(
        results,
        dd_path,
        title="Phase 4: drawdown comparison (test window)",
    )

    # Compare new strategies to the two benchmarks by Sharpe ratio.
    sharpe_lookup = dict(zip(metrics_df["strategy"], metrics_df["sharpe_ratio"]))
    sma_sharpe = sharpe_lookup[benchmarks[0].name]
    topk_sharpe = sharpe_lookup[benchmarks[1].name]
    bench_max = max(sma_sharpe, topk_sharpe)

    comparison = []
    for s in new_strategies:
        sr = sharpe_lookup[s.name]
        comparison.append({
            "strategy": s.name,
            "sharpe": sr,
            "beats_sma": bool(sr > sma_sharpe),
            "beats_topk": bool(sr > topk_sharpe),
            "beats_both_benchmarks": bool(sr > bench_max),
        })

    log_path = LOG_DIR / "new_strategies_log.json"
    log_payload = {
        "n_days": int(len(prices)),
        "n_stocks": int(prices.shape[1]),
        "train_window": {
            "start": str(start.date()),
            "end": str(train_end.date()),
        },
        "test_window": {
            "start": str(test_start.date()),
            "end": str(prices.index[-1].date()),
        },
        "benchmarks": {
            "SMACrossoverPortfolioStrategy": {
                "short_window": 20,
                "long_window": 50,
                "test_sharpe": sma_sharpe,
            },
            "TopKMomentumStrategy": {
                "lookback": 30,
                "k": 10,
                "test_sharpe": topk_sharpe,
            },
        },
        "tuned_params": {
            "ConcentratedMomentumStrategy": concentrated_params,
            "MomentumWithDrawdownControlStrategy": drawdown_params,
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
    print(f"  Benchmark 1 (SMA crossover) Sharpe = {sma_sharpe:.4f}")
    print(f"  Benchmark 2 (Top-K momentum) Sharpe = {topk_sharpe:.4f}")
    print(f"  -> Bar to clear (max benchmark) = {bench_max:.4f}")
    for row in comparison:
        verdict = "BEATS BOTH" if row["beats_both_benchmarks"] else "does NOT beat both"
        print(
            f"  {row['strategy']}: Sharpe={row['sharpe']:.4f} "
            f"-> {verdict} "
            f"(beats SMA: {row['beats_sma']}, beats Top-K: {row['beats_topk']})"
        )

    print(f"\n[phase4] Wrote: {metrics_path}")
    print(f"[phase4] Wrote: {nav_path}")
    print(f"[phase4] Wrote: {dd_path}")
    print(f"[phase4] Wrote: {log_path}")


if __name__ == "__main__":
    main()
