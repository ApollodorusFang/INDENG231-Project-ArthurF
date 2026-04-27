"""Performance metrics for daily-frequency backtest results.

All functions accept either a :class:`pandas.Series` of daily simple
returns or a NAV series, depending on what they need.  The helper
:func:`summarize_performance` collects the standard set into a flat dict.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR


def cumulative_return(daily_returns: pd.Series) -> float:
    """Total compounded return over the period."""
    if daily_returns is None or len(daily_returns) == 0:
        return 0.0
    return float((1.0 + daily_returns).prod() - 1.0)


def annualized_return(
    daily_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Geometric annualized return."""
    n = len(daily_returns)
    if n == 0:
        return 0.0
    total = float((1.0 + daily_returns).prod())
    if total <= 0:
        return -1.0
    return total ** (periods_per_year / n) - 1.0


def annualized_volatility(
    daily_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized standard deviation of daily returns (sample, ddof=1)."""
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio with ``risk_free`` quoted as an annual rate."""
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - risk_free / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def compute_drawdown_series(nav: pd.Series) -> pd.Series:
    """Return the per-period drawdown series ``nav / running_max - 1``."""
    if nav is None or len(nav) == 0:
        return pd.Series(dtype=float)
    rolling_max = nav.cummax()
    return nav / rolling_max - 1.0


def max_drawdown(nav: pd.Series) -> float:
    """Worst peak-to-trough drop expressed as a (negative) fraction."""
    dd = compute_drawdown_series(nav)
    if dd.empty:
        return 0.0
    return float(dd.min())


def win_rate(daily_returns: pd.Series) -> float:
    """Fraction of non-zero return days that are strictly positive."""
    if daily_returns is None or len(daily_returns) == 0:
        return 0.0
    nz = daily_returns[daily_returns != 0]
    if len(nz) == 0:
        return 0.0
    return float((nz > 0).mean())


def calmar_ratio(
    daily_returns: pd.Series,
    nav: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized return divided by the absolute max drawdown."""
    mdd = abs(max_drawdown(nav))
    if mdd == 0:
        return 0.0
    return annualized_return(daily_returns, periods_per_year) / mdd


def average_turnover(turnover: pd.Series) -> float:
    """Mean L1 turnover per trading day."""
    if turnover is None or len(turnover) == 0:
        return 0.0
    return float(turnover.mean())


def summarize_performance(result: Any) -> Dict[str, Any]:
    """Roll up the standard metrics into a single dictionary.

    ``result`` is expected to expose the attributes produced by
    :class:`src.backtester.BacktestResult` (``daily_returns``, ``nav``,
    ``turnover``, ``strategy_name``).
    """
    r = result.daily_returns
    return {
        "strategy": result.strategy_name,
        "cumulative_return": cumulative_return(r),
        "annualized_return": annualized_return(r),
        "annualized_volatility": annualized_volatility(r),
        "sharpe_ratio": sharpe_ratio(r),
        "max_drawdown": max_drawdown(result.nav),
        "calmar_ratio": calmar_ratio(r, result.nav),
        "win_rate": win_rate(r),
        "average_turnover": average_turnover(result.turnover),
    }
