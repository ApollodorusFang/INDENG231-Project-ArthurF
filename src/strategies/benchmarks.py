"""Benchmark portfolio strategies for Phase 3.

These cross-sectional strategies take the full Nasdaq-100 universe,
build a daily signal using only data through ``t``, select a subset of
stocks, and then convert the selection into a weight vector via the
helpers in :mod:`src.portfolio`.

All three strategies inherit from :class:`BaseStrategy` and pass their
output through :py:meth:`BaseStrategy.validate_weights`, which is what
guarantees no shorts / no leverage / cash residual at the engine level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..portfolio import equal_weight, inverse_volatility_weight
from .base import BaseStrategy


def _trailing_return(prices: pd.DataFrame, lookback: int) -> pd.Series:
    """Cross-sectional trailing return over the last ``lookback`` rows."""
    if len(prices) <= lookback:
        return pd.Series(dtype="float64")
    last = prices.iloc[-1]
    base = prices.iloc[-1 - lookback]
    ret = last / base - 1.0
    return ret.replace([np.inf, -np.inf], np.nan).dropna()


def _rolling_volatility(returns: pd.DataFrame, window: int) -> pd.Series:
    """Per-ticker daily volatility estimated on the last ``window`` rows."""
    if len(returns) < window:
        return pd.Series(dtype="float64")
    vol = returns.iloc[-window:].std(ddof=1)
    return vol.replace([np.inf, -np.inf], np.nan).dropna()


class SMACrossoverPortfolioStrategy(BaseStrategy):
    """Equal-weight every stock whose short SMA is above its long SMA."""

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        if short_window >= long_window:
            raise ValueError(
                "short_window must be strictly less than long_window."
            )
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.name = (
            f"sma_crossover_portfolio_s{self.short_window}_l{self.long_window}"
        )

    def generate_weights(
        self,
        prices_until_t: pd.DataFrame,
        returns_until_t: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        if len(prices_until_t) < self.long_window:
            return pd.Series(dtype="float64")
        short_ma = prices_until_t.iloc[-self.short_window :].mean()
        long_ma = prices_until_t.iloc[-self.long_window :].mean()
        diff = (short_ma - long_ma).replace([np.inf, -np.inf], np.nan).dropna()
        selected = list(diff[diff > 0].index)
        return equal_weight(selected, list(prices_until_t.columns))


class TopKMomentumStrategy(BaseStrategy):
    """Equal-weight the top-K stocks by trailing ``lookback`` return."""

    def __init__(self, lookback: int = 30, k: int = 10) -> None:
        if k <= 0:
            raise ValueError("k must be positive.")
        self.lookback = int(lookback)
        self.k = int(k)
        self.name = f"top{self.k}_momentum_lb{self.lookback}"

    def generate_weights(
        self,
        prices_until_t: pd.DataFrame,
        returns_until_t: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        ret = _trailing_return(prices_until_t, self.lookback)
        if ret.empty:
            return pd.Series(dtype="float64")
        top = ret.sort_values(ascending=False).head(self.k)
        return equal_weight(list(top.index), list(prices_until_t.columns))


class RiskAdjustedTopKMomentumStrategy(BaseStrategy):
    """Top-K by ``return / volatility`` signal, weighted inverse-vol."""

    def __init__(
        self,
        lookback: int = 30,
        volatility_window: int = 20,
        k: int = 10,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive.")
        self.lookback = int(lookback)
        self.volatility_window = int(volatility_window)
        self.k = int(k)
        self.name = (
            f"risk_adj_top{self.k}_momentum"
            f"_lb{self.lookback}_vw{self.volatility_window}"
        )

    def generate_weights(
        self,
        prices_until_t: pd.DataFrame,
        returns_until_t: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        ret = _trailing_return(prices_until_t, self.lookback)
        vol = _rolling_volatility(returns_until_t, self.volatility_window)
        if ret.empty or vol.empty:
            return pd.Series(dtype="float64")

        common = ret.index.intersection(vol.index)
        if len(common) == 0:
            return pd.Series(dtype="float64")
        signal = (ret.loc[common] / vol.loc[common]).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        signal = signal[signal > 0]
        if signal.empty:
            return pd.Series(dtype="float64")

        top = signal.sort_values(ascending=False).head(self.k)
        return inverse_volatility_weight(
            list(top.index),
            vol.loc[top.index],
            list(prices_until_t.columns),
        )
