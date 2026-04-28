"""Portfolio-construction helpers used by Phase 3 strategies.

Each helper turns a set of selected tickers (plus optional auxiliary
information such as rolling volatility or a cross-sectional signal)
into a full-universe weight vector that satisfies the project's hard
constraints:

* weights >= 0 (no short selling),
* sum of weights <= 1 (no leverage; the residual is implicit cash),
* cash-only when no ticker qualifies.

The returned series is always reindexed to ``universe`` so callers can
hand it straight to :py:meth:`BaseStrategy.validate_weights` without
extra plumbing.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _full_universe_zeros(universe: Iterable[str]) -> pd.Series:
    return pd.Series(0.0, index=list(universe))


def equal_weight(
    selected_tickers: Sequence[str],
    universe: Iterable[str],
) -> pd.Series:
    """Spread capital evenly across ``selected_tickers``.

    Tickers not in ``universe`` are dropped before the split.  An empty
    selection yields an all-cash (all-zero) vector.
    """
    weights = _full_universe_zeros(universe)
    valid = [t for t in selected_tickers if t in weights.index]
    n = len(valid)
    if n == 0:
        return weights
    weights.loc[valid] = 1.0 / n
    return weights


def inverse_volatility_weight(
    selected_tickers: Sequence[str],
    rolling_volatility: pd.Series,
    universe: Iterable[str],
) -> pd.Series:
    """Allocate proportional to ``1 / volatility`` across selected tickers.

    Tickers whose volatility is zero, NaN, or +-inf are dropped.  If no
    tickers survive, the result is all cash.
    """
    weights = _full_universe_zeros(universe)
    if rolling_volatility is None or len(rolling_volatility) == 0:
        return weights

    vol = pd.Series(rolling_volatility, dtype="float64")
    vol = vol.reindex([t for t in selected_tickers if t in weights.index])
    vol = vol.replace([np.inf, -np.inf], np.nan).dropna()
    vol = vol[vol > 0]
    if vol.empty:
        return weights

    inv = 1.0 / vol
    inv_sum = float(inv.sum())
    if inv_sum <= 0 or not np.isfinite(inv_sum):
        return weights
    weights.loc[inv.index] = (inv / inv_sum).values
    return weights


def rank_weight(
    signal: pd.Series,
    selected_tickers: Sequence[str],
    universe: Iterable[str],
) -> pd.Series:
    """Rank-based weighting -- higher ``signal`` -> higher weight.

    Among ``selected_tickers``, drop entries whose signal is NaN or
    non-finite, then assign weights proportional to ``rank``
    (1, 2, ..., n).  An empty surviving set yields all cash.
    """
    weights = _full_universe_zeros(universe)
    if signal is None or len(signal) == 0:
        return weights

    sig = pd.Series(signal, dtype="float64")
    sig = sig.reindex([t for t in selected_tickers if t in weights.index])
    sig = sig.replace([np.inf, -np.inf], np.nan).dropna()
    if sig.empty:
        return weights

    # `rank` returns 1 for the smallest, n for the largest -- which means
    # the highest-signal name receives the largest weight, matching the
    # required behavior.
    ranks = sig.rank(method="average")
    total = float(ranks.sum())
    if total <= 0 or not np.isfinite(total):
        return weights
    weights.loc[ranks.index] = (ranks / total).values
    return weights
