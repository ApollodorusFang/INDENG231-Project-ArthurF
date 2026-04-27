"""Base classes and helpers for daily-close trading strategies.

A strategy maps the information available at the close of day ``t`` -- prices
and returns up to and including ``t`` -- to a non-negative weight vector that
sums to at most 1.  The backtester applies those weights to the realized
returns of day ``t+1``, which is what avoids look-ahead bias.

Subclasses only need to override :py:meth:`BaseStrategy.generate_weights`.
The shared :py:meth:`BaseStrategy.validate_weights` enforces the project's
hard constraints (no shorts, no leverage, full alignment to the universe).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Numerical slack for sum-to-one and non-negativity assertions.
_TOLERANCE = 1e-9


class BaseStrategy:
    """Abstract base class for all backtesting strategies."""

    name: str = "base"

    def generate_weights(
        self,
        prices_until_t: pd.DataFrame,
        returns_until_t: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        """Produce target portfolio weights for ``current_date``.

        Parameters
        ----------
        prices_until_t : pd.DataFrame
            Wide price panel sliced to include only data through (and
            including) ``current_date``.  No future prices are exposed.
        returns_until_t : pd.DataFrame
            Wide simple-return panel matching ``prices_until_t``.
        current_date : pd.Timestamp
            The trading day whose close is the decision point.

        Returns
        -------
        pd.Series
            Raw target weights indexed by ticker.  May contain a subset
            of the universe; missing tickers are treated as zero.  The
            backtester normalizes via :py:meth:`validate_weights`.
        """
        raise NotImplementedError

    def validate_weights(
        self,
        weights: pd.Series,
        universe: Iterable[str],
    ) -> pd.Series:
        """Align weights to the full universe and enforce constraints.

        The returned series:

        * is reindexed to ``universe`` (missing tickers become 0);
        * has NaN / +-inf entries replaced with 0;
        * has any negative entries clipped to 0 (no short selling);
        * is scaled down proportionally if the total exceeds 1
          (no leverage); the residual is implicit cash.
        """
        universe = list(universe)
        if weights is None or (isinstance(weights, pd.Series) and weights.empty):
            return pd.Series(0.0, index=universe)

        w = pd.Series(weights, dtype="float64")
        w = w.reindex(universe)
        w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        w = w.clip(lower=0.0)

        total = float(w.sum())
        if total > 1.0:
            w = w / total

        # Numerical hygiene before the assertions.
        w = w.clip(lower=0.0)
        assert (w.values >= -_TOLERANCE).all(), "Weights must be non-negative."
        assert (
            float(w.sum()) <= 1.0 + _TOLERANCE
        ), f"Sum of weights exceeds 1: {float(w.sum())!r}"
        return w


class EqualWeightBuyAndHoldStrategy(BaseStrategy):
    """Equal-weight buy-and-hold over every available stock.

    On the first day the strategy can act, it spreads capital equally
    across the tickers that have a valid price.  After that the position
    is held: the *target* weights drift with realized prices, exactly as
    a real buy-and-hold portfolio would.  This is enough for a smoke
    test of the engine.
    """

    name = "equal_weight_buy_and_hold"

    def __init__(self) -> None:
        self._initialized: bool = False
        self._initial_prices: pd.Series | None = None
        self._initial_weights: pd.Series | None = None

    def generate_weights(
        self,
        prices_until_t: pd.DataFrame,
        returns_until_t: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        latest = prices_until_t.iloc[-1].dropna()

        if not self._initialized:
            if latest.empty:
                return pd.Series(dtype=float)
            n = len(latest)
            self._initial_weights = pd.Series(1.0 / n, index=latest.index)
            self._initial_prices = latest.copy()
            self._initialized = True
            return self._initial_weights

        # Drift weights with cumulative price ratios since inception.
        held = self._initial_prices.index
        cur = prices_until_t.iloc[-1].reindex(held)
        cur = cur.fillna(self._initial_prices)  # fall back if a price went NaN
        notional = self._initial_weights * (cur / self._initial_prices)
        total = float(notional.sum())
        if total <= 0:
            return self._initial_weights
        return notional / total
