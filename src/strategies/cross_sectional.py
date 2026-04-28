"""Phase 4 cross-sectional strategies.

Two new portfolio strategies designed to beat the Phase 3 benchmarks
(``SMACrossoverPortfolioStrategy``, ``TopKMomentumStrategy``) on Sharpe
ratio:

* :class:`RiskAdjustedMomentumStrategy` -- top-K by ``return / vol``,
  with an optional market-volatility filter that scales gross exposure
  down when realized market vol spikes.
* :class:`LowVolatilityMomentumStrategy` -- top-K by a transparent
  combination of trailing-return rank and low-volatility rank.

Both strategies use only information through day ``t`` and route their
weights through :py:meth:`BaseStrategy.validate_weights`, which enforces
the project's no-short / no-leverage / cash-residual contract.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..portfolio import inverse_volatility_weight
from .base import BaseStrategy


def _trailing_return(prices: pd.DataFrame, lookback: int) -> pd.Series:
    if len(prices) <= lookback:
        return pd.Series(dtype="float64")
    last = prices.iloc[-1]
    base = prices.iloc[-1 - lookback]
    ret = last / base - 1.0
    return ret.replace([np.inf, -np.inf], np.nan).dropna()


def _rolling_volatility(returns: pd.DataFrame, window: int) -> pd.Series:
    if len(returns) < window:
        return pd.Series(dtype="float64")
    vol = returns.iloc[-window:].std(ddof=1)
    return vol.replace([np.inf, -np.inf], np.nan).dropna()


def _market_volatility(returns: pd.DataFrame, window: int) -> float:
    """Realized volatility of the equal-weight market over ``window`` days."""
    if len(returns) < window:
        return float("nan")
    market = returns.iloc[-window:].mean(axis=1)
    if len(market) < 2:
        return float("nan")
    return float(market.std(ddof=1))


class RiskAdjustedMomentumStrategy(BaseStrategy):
    """Top-K return/vol momentum with an optional market-vol throttle."""

    def __init__(
        self,
        lookback: int = 60,
        volatility_window: int = 20,
        k: int = 10,
        market_vol_window: int = 20,
        market_vol_threshold: Optional[float] = None,
        defensive_exposure: float = 0.5,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive.")
        if not 0.0 <= defensive_exposure <= 1.0:
            raise ValueError("defensive_exposure must lie in [0, 1].")
        self.lookback = int(lookback)
        self.volatility_window = int(volatility_window)
        self.k = int(k)
        self.market_vol_window = int(market_vol_window)
        self.market_vol_threshold = market_vol_threshold
        self.defensive_exposure = float(defensive_exposure)
        thr = (
            f"{self.market_vol_threshold:g}"
            if self.market_vol_threshold is not None
            else "off"
        )
        self.name = (
            f"risk_adj_momentum_lb{self.lookback}_vw{self.volatility_window}"
            f"_k{self.k}_mvw{self.market_vol_window}_mvt{thr}"
            f"_def{self.defensive_exposure:g}"
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
        signal = (
            (ret.loc[common] / vol.loc[common])
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        signal = signal[signal > 0]
        if signal.empty:
            return pd.Series(dtype="float64")

        top = signal.sort_values(ascending=False).head(self.k)
        weights = inverse_volatility_weight(
            list(top.index),
            vol.loc[top.index],
            list(prices_until_t.columns),
        )

        # Optional market-vol throttle: when realized market vol exceeds
        # the threshold, dial gross exposure down to defensive_exposure
        # and let validate_weights treat the remainder as cash.
        if self.market_vol_threshold is not None:
            mkt_vol = _market_volatility(returns_until_t, self.market_vol_window)
            if np.isfinite(mkt_vol) and mkt_vol > self.market_vol_threshold:
                weights = weights * self.defensive_exposure

        return weights


class LowVolatilityMomentumStrategy(BaseStrategy):
    """Top-K by combined trailing-return rank + low-volatility rank."""

    def __init__(
        self,
        lookback: int = 60,
        volatility_window: int = 20,
        k: int = 10,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive.")
        self.lookback = int(lookback)
        self.volatility_window = int(volatility_window)
        self.k = int(k)
        self.name = (
            f"low_vol_momentum_lb{self.lookback}"
            f"_vw{self.volatility_window}_k{self.k}"
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
        ret = ret.loc[common]
        vol = vol.loc[common]

        # Keep only positive-return, finite-volatility names.
        mask = (ret > 0) & np.isfinite(vol) & (vol > 0)
        ret = ret[mask]
        vol = vol[mask]
        if ret.empty:
            return pd.Series(dtype="float64")

        # Higher return -> larger return rank; lower vol -> larger
        # low-vol rank.  Sum is a transparent combined score.
        return_rank = ret.rank(method="average", ascending=True)
        low_vol_rank = vol.rank(method="average", ascending=False)
        score = return_rank + low_vol_rank

        top = score.sort_values(ascending=False).head(self.k)
        return inverse_volatility_weight(
            list(top.index),
            vol.loc[top.index],
            list(prices_until_t.columns),
        )
