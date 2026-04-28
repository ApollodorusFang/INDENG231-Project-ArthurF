"""Single-stock long-or-cash strategies.

Each strategy targets exactly one ticker.  When its signal fires the
portfolio is fully invested in that ticker (weight 1.0); otherwise it
holds 100% cash (weight 0.0).  The strategies share three guarantees:

* They use only price/return information up to and including ``t``.
* They return a partial weight series (just the target ticker), and the
  base class :py:meth:`BaseStrategy.validate_weights` reindexes it to
  the full universe with zeros elsewhere.
* When there is not enough history to evaluate the signal, they hold
  cash.

No ticker names are baked in -- the symbol is a constructor argument.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy


def _full_weight(ticker: str) -> pd.Series:
    """Return a Series allocating 100% to ``ticker``."""
    return pd.Series({ticker: 1.0}, dtype="float64")


def _cash() -> pd.Series:
    """Return an empty Series -- the validator interprets this as all-cash."""
    return pd.Series(dtype="float64")


def _ticker_history(prices: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    """Pull the price history for ``ticker`` if it exists in the panel."""
    if ticker not in prices.columns:
        return None
    s = prices[ticker].dropna()
    return s if not s.empty else None


class SingleStockMomentumStrategy(BaseStrategy):
    """Go long a single stock when its trailing return is positive."""

    def __init__(self, ticker: str, lookback: int = 20) -> None:
        self.ticker = ticker
        self.lookback = int(lookback)
        self.name = f"momentum_{ticker}_lb{self.lookback}"

    def generate_weights(
        self,
        prices_until_t: pd.DataFrame,
        returns_until_t: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        hist = _ticker_history(prices_until_t, self.ticker)
        if hist is None or len(hist) <= self.lookback:
            return _cash()
        trailing_ret = hist.iloc[-1] / hist.iloc[-1 - self.lookback] - 1.0
        if trailing_ret > 0:
            return _full_weight(self.ticker)
        return _cash()


class SingleStockMeanReversionStrategy(BaseStrategy):
    """Go long a single stock after a sharp recent drawdown."""

    def __init__(
        self,
        ticker: str,
        lookback: int = 20,
        threshold: float = -0.05,
    ) -> None:
        self.ticker = ticker
        self.lookback = int(lookback)
        self.threshold = float(threshold)
        self.name = (
            f"mean_reversion_{ticker}_lb{self.lookback}_thr{self.threshold:g}"
        )

    def generate_weights(
        self,
        prices_until_t: pd.DataFrame,
        returns_until_t: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        hist = _ticker_history(prices_until_t, self.ticker)
        if hist is None or len(hist) <= self.lookback:
            return _cash()
        trailing_ret = hist.iloc[-1] / hist.iloc[-1 - self.lookback] - 1.0
        if trailing_ret < self.threshold:
            return _full_weight(self.ticker)
        return _cash()


class MovingAverageCrossoverStrategy(BaseStrategy):
    """Long when the short SMA is above the long SMA, else cash."""

    def __init__(
        self,
        ticker: str,
        short_window: int = 20,
        long_window: int = 50,
    ) -> None:
        if short_window >= long_window:
            raise ValueError(
                "short_window must be strictly less than long_window."
            )
        self.ticker = ticker
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.name = (
            f"ma_crossover_{ticker}_s{self.short_window}_l{self.long_window}"
        )

    def generate_weights(
        self,
        prices_until_t: pd.DataFrame,
        returns_until_t: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        hist = _ticker_history(prices_until_t, self.ticker)
        if hist is None or len(hist) < self.long_window:
            return _cash()
        short_ma = hist.iloc[-self.short_window :].mean()
        long_ma = hist.iloc[-self.long_window :].mean()
        if short_ma > long_ma:
            return _full_weight(self.ticker)
        return _cash()


class VolatilityFilteredMomentumStrategy(BaseStrategy):
    """Momentum signal that only fires when recent volatility is calm."""

    def __init__(
        self,
        ticker: str,
        lookback: int = 20,
        volatility_window: int = 20,
        volatility_threshold: float = 0.03,
    ) -> None:
        self.ticker = ticker
        self.lookback = int(lookback)
        self.volatility_window = int(volatility_window)
        self.volatility_threshold = float(volatility_threshold)
        self.name = (
            f"vol_filt_momentum_{ticker}_lb{self.lookback}"
            f"_vw{self.volatility_window}_vt{self.volatility_threshold:g}"
        )

    def generate_weights(
        self,
        prices_until_t: pd.DataFrame,
        returns_until_t: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        hist = _ticker_history(prices_until_t, self.ticker)
        if hist is None or len(hist) <= max(self.lookback, self.volatility_window):
            return _cash()
        trailing_ret = hist.iloc[-1] / hist.iloc[-1 - self.lookback] - 1.0
        if self.ticker not in returns_until_t.columns:
            return _cash()
        rets = returns_until_t[self.ticker].dropna()
        if len(rets) < self.volatility_window:
            return _cash()
        vol = float(rets.iloc[-self.volatility_window :].std(ddof=1))
        if np.isnan(vol):
            return _cash()
        if trailing_ret > 0 and vol < self.volatility_threshold:
            return _full_weight(self.ticker)
        return _cash()


class ZScoreMeanReversionStrategy(BaseStrategy):
    """Long when the price's rolling z-score drops below ``z_threshold``."""

    def __init__(
        self,
        ticker: str,
        window: int = 20,
        z_threshold: float = -1.0,
    ) -> None:
        self.ticker = ticker
        self.window = int(window)
        self.z_threshold = float(z_threshold)
        self.name = (
            f"zscore_mean_reversion_{ticker}_w{self.window}_z{self.z_threshold:g}"
        )

    def generate_weights(
        self,
        prices_until_t: pd.DataFrame,
        returns_until_t: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        hist = _ticker_history(prices_until_t, self.ticker)
        if hist is None or len(hist) < self.window:
            return _cash()
        window_slice = hist.iloc[-self.window :]
        mu = float(window_slice.mean())
        sd = float(window_slice.std(ddof=1))
        if sd == 0 or np.isnan(sd):
            return _cash()
        z = (float(hist.iloc[-1]) - mu) / sd
        if z < self.z_threshold:
            return _full_weight(self.ticker)
        return _cash()
