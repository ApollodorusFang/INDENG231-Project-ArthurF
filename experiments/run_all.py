"""Phase 1 entry point: smoke-test the engine on equal-weight buy-and-hold.

Run from the repository root:

    python experiments/run_all.py

Outputs:
    outputs/tables/phase1_metrics.csv
    outputs/figures/phase1_nav.png
    outputs/figures/phase1_drawdown.png
    outputs/logs/phase1_log.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Make `python experiments/run_all.py` work from the repo root by ensuring
# the project root is on sys.path before importing `src`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.backtester import Backtester  # noqa: E402
from src.config import DATA_PATH, FIGURE_DIR, LOG_DIR, TABLE_DIR  # noqa: E402
from src.data_loader import compute_returns, load_price_data  # noqa: E402
from src.metrics import summarize_performance  # noqa: E402
from src.plotting import plot_drawdown, plot_nav  # noqa: E402
from src.strategies.base import EqualWeightBuyAndHoldStrategy  # noqa: E402


def main() -> None:
    print(f"[phase1] Loading price data from {DATA_PATH}")
    prices = load_price_data(DATA_PATH)
    returns = compute_returns(prices)
    print(
        f"[phase1] Loaded {prices.shape[1]} stocks across "
        f"{len(prices)} trading days "
        f"({prices.index.min().date()} -> {prices.index.max().date()})"
    )

    strategy = EqualWeightBuyAndHoldStrategy()
    bt = Backtester(prices, returns=returns)
    result = bt.run(strategy)

    metrics = summarize_performance(result)
    print("\n=== Phase 1 metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:>22}: {v:.6f}")
        else:
            print(f"  {k:>22}: {v}")

    metrics_path = TABLE_DIR / "phase1_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    nav_path = FIGURE_DIR / "phase1_nav.png"
    dd_path = FIGURE_DIR / "phase1_drawdown.png"
    plot_nav(result, nav_path)
    plot_drawdown(result, dd_path)

    log_path = LOG_DIR / "phase1_log.json"
    log_payload = {
        "strategy": result.strategy_name,
        "metrics": metrics,
        "n_days": int(len(result.nav)),
        "n_stocks": int(prices.shape[1]),
        "start": str(result.nav.index.min().date()),
        "end": str(result.nav.index.max().date()),
        "initial_capital": result.initial_capital,
        "transaction_cost": result.transaction_cost,
    }
    with open(log_path, "w") as f:
        json.dump(log_payload, f, indent=2)

    print(f"\n[phase1] Wrote: {metrics_path}")
    print(f"[phase1] Wrote: {nav_path}")
    print(f"[phase1] Wrote: {dd_path}")
    print(f"[phase1] Wrote: {log_path}")


if __name__ == "__main__":
    main()
