"""Phase 4 cross-sectional strategies.

Four portfolio strategies that aim to beat the Phase 3 benchmarks
(``SMACrossoverPortfolioStrategy``, ``TopKMomentumStrategy``) on Sharpe
ratio:

* :class:`RiskAdjustedMomentumStrategy` -- top-K by ``return / vol``,
  with an optional market-volatility filter that scales gross exposure
  down when realized market vol spikes.
* :class:`LowVolatilityMomentumStrategy` -- top-K by a transparent
  combination of trailing-return rank and low-volatility rank.
* :class:`ConcentratedMomentumStrategy` -- top-K by raw trailing return,
  optionally inverse-volatility weighted, with positive-signal gating.
* :class:`MomentumWithDrawdownControlStrategy` -- top-K momentum with a
  market-drawdown overlay that scales gross exposure down (or to cash)
  when the equal-weight market index draws down beyond a threshold.

All four strategies use only information through day ``t`` and route
their weights through :py:meth:`BaseStrategy.validate_weights`, which
enforces the project's no-short / no-leverage / cash-residual contract.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..portfolio import equal_weight, inverse_volatility_weight
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


class ConcentratedMomentumStrategy(BaseStrategy):
    """Top-K trailing-return momentum, equal- or inverse-vol-weighted.

    Concentration plus optional inverse-volatility weighting is meant to
    keep the strongest part of the raw-momentum signal (which is what
    beat the Phase 4 benchmarks in the held-out test window) while
    controlling realized volatility.  The signal is the same trailing
    cross-sectional return ranking used by ``TopKMomentumStrategy``;
    we additionally require the selected name to have a *strictly
    positive* trailing return so the portfolio sits in cash during
    broad-market down phases.
    """

    def __init__(
        self,
        lookback: int = 30,
        k: int = 5,
        weighting: str = "inverse_vol",
        volatility_window: int = 20,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive.")
        if weighting not in {"equal", "inverse_vol"}:
            raise ValueError(
                "weighting must be 'equal' or 'inverse_vol'."
            )
        self.lookback = int(lookback)
        self.k = int(k)
        self.weighting = weighting
        self.volatility_window = int(volatility_window)
        self.name = (
            f"concentrated_momentum_lb{self.lookback}_k{self.k}"
            f"_{self.weighting}_vw{self.volatility_window}"
        )

    def generate_weights(
        self,
        prices_until_t: pd.DataFrame,
        returns_until_t: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        ret = _trailing_return(prices_until_t, self.lookback)
        if ret.empty:
            return pd.Series(dtype="float64")

        # Positive-momentum gate.  When the cross section has nothing
        # positive, we hold cash.  This protects against bear regimes.
        positive = ret[ret > 0]
        if positive.empty:
            return pd.Series(dtype="float64")

        top = positive.sort_values(ascending=False).head(self.k)

        if self.weighting == "equal":
            return equal_weight(list(top.index), list(prices_until_t.columns))

        vol = _rolling_volatility(returns_until_t, self.volatility_window)
        if vol.empty:
            # Fall back to equal weight if vol is not yet estimable.
            return equal_weight(list(top.index), list(prices_until_t.columns))
        return inverse_volatility_weight(
            list(top.index),
            vol.reindex(top.index),
            list(prices_until_t.columns),
        )


def _market_drawdown(
    returns: pd.DataFrame,
    window: int,
) -> float:
    """Realized drawdown of the equal-weight market index over ``window``.

    Uses only data through the most recent ``window`` rows of ``returns``
    so it is safe to call from inside ``generate_weights`` on the
    ``returns_until_t`` slice.
    """
    if returns is None or len(returns) < 2:
        return 0.0
    w = min(int(window), len(returns))
    market_returns = returns.iloc[-w:].mean(axis=1)
    if market_returns.empty:
        return 0.0
    nav = (1.0 + market_returns).cumprod()
    running_max = nav.cummax()
    return float(nav.iloc[-1] / running_max.iloc[-1] - 1.0)


class MomentumWithDrawdownControlStrategy(BaseStrategy):
    """Top-K momentum with an equal-weight-market drawdown throttle.

    Selection: top-K cross-sectional names by trailing return
    (positive-only), equal-weighted by default.  Risk overlay: track the
    realized drawdown of the equal-weight market index over the last
    ``market_window`` days; if that drawdown is more negative than
    ``drawdown_threshold``, scale gross exposure down to
    ``defensive_exposure`` (and the residual sits in cash).

    The drawdown signal uses only past returns through ``t`` (no
    look-ahead).  It is a transparent regime filter -- when the broad
    market is in a sustained down move, we stop fully chasing momentum.
    """

    def __init__(
        self,
        lookback: int = 30,
        k: int = 10,
        drawdown_threshold: float = -0.10,
        defensive_exposure: float = 0.5,
        market_window: int = 120,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive.")
        if not 0.0 <= defensive_exposure <= 1.0:
            raise ValueError("defensive_exposure must lie in [0, 1].")
        if drawdown_threshold > 0:
            raise ValueError("drawdown_threshold must be <= 0.")
        self.lookback = int(lookback)
        self.k = int(k)
        self.drawdown_threshold = float(drawdown_threshold)
        self.defensive_exposure = float(defensive_exposure)
        self.market_window = int(market_window)
        self.name = (
            f"momentum_dd_ctrl_lb{self.lookback}_k{self.k}"
            f"_thr{self.drawdown_threshold:g}"
            f"_def{self.defensive_exposure:g}"
            f"_mw{self.market_window}"
        )

    def generate_weights(
        self,
        prices_until_t: pd.DataFrame,
        returns_until_t: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        ret = _trailing_return(prices_until_t, self.lookback)
        if ret.empty:
            return pd.Series(dtype="float64")

        positive = ret[ret > 0]
        if positive.empty:
            return pd.Series(dtype="float64")

        top = positive.sort_values(ascending=False).head(self.k)
        weights = equal_weight(
            list(top.index), list(prices_until_t.columns)
        )

        mkt_dd = _market_drawdown(returns_until_t, self.market_window)
        if mkt_dd < self.drawdown_threshold:
            weights = weights * self.defensive_exposure

        return weights
