"""Matplotlib plots for backtest results.

The Agg backend is forced so the helpers run headless in CI / scripted
environments.  Each function takes a :class:`BacktestResult`-like object
and a destination path, creates the parent directory if needed, and
writes a PNG.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .metrics import compute_drawdown_series

PathLike = Union[str, Path]


def plot_nav(result: Any, path: PathLike) -> Path:
    """Plot the NAV curve and save it as a PNG."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    result.nav.plot(ax=ax, color="tab:blue", linewidth=1.5)
    ax.set_title(f"NAV — {result.strategy_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"NAV (initial = {result.initial_capital:g})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_drawdown(result: Any, path: PathLike) -> Path:
    """Plot the drawdown series (filled area, in red) and save as PNG."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    dd = compute_drawdown_series(result.nav)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(dd.index, dd.values, 0.0, color="tab:red", alpha=0.4)
    ax.plot(dd.index, dd.values, color="tab:red", linewidth=1.0)
    ax.set_title(f"Drawdown — {result.strategy_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
