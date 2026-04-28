"""Matplotlib plots for backtest results.

The Agg backend is forced so the helpers run headless in CI / scripted
environments.  Each function takes a :class:`BacktestResult`-like object
and a destination path, creates the parent directory if needed, and
writes a PNG.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Union

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


def plot_multiple_nav(
    results: Mapping[str, Any],
    path: PathLike,
    title: str = "NAV comparison",
) -> Path:
    """Overlay NAV curves from multiple backtest results on a single chart.

    ``results`` maps a display name (used in the legend) to a
    BacktestResult-like object exposing ``nav``.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label, res in results.items():
        ax.plot(res.nav.index, res.nav.values, linewidth=1.4, label=label)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV (initial = 1.0)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_multiple_drawdowns(
    results: Mapping[str, Any],
    path: PathLike,
    title: str = "Drawdown comparison",
) -> Path:
    """Overlay drawdown curves from multiple backtest results on one chart."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for label, res in results.items():
        dd = compute_drawdown_series(res.nav)
        ax.plot(dd.index, dd.values, linewidth=1.2, label=label)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
