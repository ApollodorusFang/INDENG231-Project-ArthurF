"""Statistical and risk-analysis utilities for backtest results.

Everything here works on a single :class:`pandas.Series` of daily simple
returns (or a wide return panel for the optional copula simulator).  No
strategy logic lives in this module -- it only consumes realized return
streams produced upstream by the backtester, so it cannot introduce
look-ahead bias on its own.

Functions are deliberately small and transparent so they're easy to
audit in the report:

* :func:`value_at_risk` / :func:`conditional_value_at_risk` -- empirical
  VaR / CVaR at a confidence level.
* :func:`sharpe_confidence_interval` -- normal-approximation CI from
  Lo (2002).
* :func:`bootstrap_nav_paths` -- block-free i.i.d. bootstrap of NAV
  trajectories.
* :func:`estimate_tail_loss_probability` -- empirical
  P(daily return < threshold).
* :func:`importance_sampling_tail_stress` -- a transparent toy stress
  test that oversamples bad days; explicitly *not* a rigorous IS
  estimator (see docstring).
* :func:`rolling_volatility` -- annualized rolling-window vol.
* :func:`gaussian_copula_simulation` -- joint Gaussian Monte-Carlo with
  a robust covariance fallback for nearly-singular matrices.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .config import TRADING_DAYS_PER_YEAR


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _clean_returns(returns: pd.Series) -> pd.Series:
    """Drop NaN/inf entries; return as a clean float64 Series."""
    if returns is None:
        return pd.Series(dtype="float64")
    s = pd.Series(returns, dtype="float64")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


# ----------------------------------------------------------------------
# tail / risk metrics
# ----------------------------------------------------------------------

def value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical (empirical) Value-at-Risk at level ``alpha``.

    Returns a *negative* number: the alpha-quantile of the daily-return
    distribution.  ``alpha=0.05`` is the worst-5% quantile.
    """
    s = _clean_returns(returns)
    if s.empty:
        return 0.0
    return float(np.quantile(s.values, alpha))


def conditional_value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    """Expected shortfall: mean return among observations <= VaR_alpha."""
    s = _clean_returns(returns)
    if s.empty:
        return 0.0
    var_alpha = float(np.quantile(s.values, alpha))
    tail = s[s <= var_alpha]
    if tail.empty:
        return float(var_alpha)
    return float(tail.mean())


def estimate_tail_loss_probability(
    returns: pd.Series,
    loss_threshold: float = -0.05,
) -> float:
    """Empirical P(daily return < ``loss_threshold``)."""
    s = _clean_returns(returns)
    if s.empty:
        return 0.0
    return float((s < loss_threshold).mean())


def rolling_volatility(
    returns: pd.Series,
    window: int = 20,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Annualized rolling standard deviation of daily returns."""
    s = _clean_returns(returns)
    if s.empty:
        return pd.Series(dtype="float64")
    return s.rolling(window=window, min_periods=max(2, window // 2)).std(
        ddof=1
    ) * np.sqrt(periods_per_year)


# ----------------------------------------------------------------------
# Sharpe-ratio confidence interval (Lo 2002 normal approximation)
# ----------------------------------------------------------------------

def sharpe_confidence_interval(
    returns: pd.Series,
    trading_days: int = TRADING_DAYS_PER_YEAR,
    confidence: float = 0.95,
) -> dict:
    """Normal-approximation CI for the annualized Sharpe ratio.

    Uses the classical CLT-based standard error
    ``SE(SR) = sqrt((1 + 0.5 SR^2) / N)``, multiplied by
    ``sqrt(trading_days)`` to lift the daily Sharpe to an annual one.
    Returns a dict with ``sharpe``, ``standard_error``, ``lower_ci``,
    ``upper_ci``.
    """
    s = _clean_returns(returns)
    n = len(s)
    if n < 2:
        return {
            "sharpe": 0.0,
            "standard_error": 0.0,
            "lower_ci": 0.0,
            "upper_ci": 0.0,
        }

    sd = float(s.std(ddof=1))
    if sd == 0.0 or not np.isfinite(sd):
        return {
            "sharpe": 0.0,
            "standard_error": 0.0,
            "lower_ci": 0.0,
            "upper_ci": 0.0,
        }

    daily_sr = float(s.mean()) / sd
    ann_sr = daily_sr * np.sqrt(trading_days)
    se_daily = np.sqrt((1.0 + 0.5 * daily_sr ** 2) / n)
    se_ann = se_daily * np.sqrt(trading_days)

    # Two-sided z-score for the requested confidence level.
    confidence = float(np.clip(confidence, 1e-6, 1 - 1e-6))
    p = 1.0 - (1.0 - confidence) / 2.0  # upper tail
    z = float(_norm_ppf(p))

    return {
        "sharpe": ann_sr,
        "standard_error": se_ann,
        "lower_ci": ann_sr - z * se_ann,
        "upper_ci": ann_sr + z * se_ann,
    }


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF via Beasley-Springer-Moro approximation.

    Self-contained so the module does not need ``scipy``.
    """
    p = float(np.clip(p, 1e-10, 1 - 1e-10))
    a = [
        -3.969683028665376e01, 2.209460984245205e02,
        -2.759285104469687e02, 1.383577518672690e02,
        -3.066479806614716e01, 2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02,
        -1.556989798598866e02, 6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e00, -2.549732539343734e00,
        4.374664141464968e00, 2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e00, 3.754408661907416e00,
    ]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = np.sqrt(-2.0 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = np.sqrt(-2.0 * np.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


# ----------------------------------------------------------------------
# bootstrap NAV paths
# ----------------------------------------------------------------------

def bootstrap_nav_paths(
    returns: pd.Series,
    n_paths: int = 1000,
    horizon: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """I.i.d. bootstrap of NAV trajectories starting from 1.0.

    For each of ``n_paths`` paths we draw ``horizon`` returns *with
    replacement* from the empirical distribution of ``returns`` and
    cumulate them.  The result is a DataFrame of shape
    ``(horizon + 1, n_paths)`` whose first row is all 1.0.
    """
    s = _clean_returns(returns)
    if s.empty:
        return pd.DataFrame()

    h = int(horizon) if horizon is not None else len(s)
    if h <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(int(seed))
    samples = rng.choice(s.values, size=(h, int(n_paths)), replace=True)
    growth = 1.0 + samples
    nav = np.cumprod(growth, axis=0)
    nav = np.vstack([np.ones((1, int(n_paths))), nav])
    cols = [f"path_{i:04d}" for i in range(int(n_paths))]
    return pd.DataFrame(nav, columns=cols)


# ----------------------------------------------------------------------
# educational tail-stress test
# ----------------------------------------------------------------------

def importance_sampling_tail_stress(
    returns: pd.Series,
    loss_threshold: float = -0.05,
    stress_multiplier: float = 3.0,
    seed: int = 42,
    n_samples: int = 10_000,
) -> dict:
    """Educational tail-stress test that oversamples bad-return days.

    .. note::
       This is an *educational* stress test, **not** a production-grade
       importance-sampling estimator.  No likelihood-ratio reweighting
       is applied.  We simply assign a higher draw probability to days
       on which the realized return was negative (multiplied by
       ``stress_multiplier``), then re-estimate the tail-loss probability
       and CVaR on a Monte-Carlo resample.  The intent is to ask
       "what if bad days happened ``stress_multiplier``-times more
       often" -- a transparent worst-case sanity check.

    Returns a dict with ``stressed_tail_probability``, ``stressed_cvar``,
    ``baseline_tail_probability``, and ``baseline_cvar``.
    """
    s = _clean_returns(returns)
    if s.empty:
        return {
            "stressed_tail_probability": 0.0,
            "stressed_cvar": 0.0,
            "baseline_tail_probability": 0.0,
            "baseline_cvar": 0.0,
        }

    base_tail = float((s < loss_threshold).mean())
    base_cvar = conditional_value_at_risk(s, alpha=0.05)

    weights = np.where(s.values < 0, float(stress_multiplier), 1.0)
    if weights.sum() == 0:
        # Degenerate case: no negative observations.  Fall back to uniform.
        weights = np.ones_like(weights)
    probs = weights / weights.sum()

    rng = np.random.default_rng(int(seed))
    draws = rng.choice(s.values, size=int(n_samples), replace=True, p=probs)

    stressed_tail = float((draws < loss_threshold).mean())
    var_threshold = float(np.quantile(draws, 0.05))
    stressed_tail_returns = draws[draws <= var_threshold]
    stressed_cvar = (
        float(stressed_tail_returns.mean())
        if stressed_tail_returns.size
        else float(var_threshold)
    )

    return {
        "stressed_tail_probability": stressed_tail,
        "stressed_cvar": stressed_cvar,
        "baseline_tail_probability": base_tail,
        "baseline_cvar": base_cvar,
    }


# ----------------------------------------------------------------------
# optional Gaussian copula Monte-Carlo
# ----------------------------------------------------------------------

def gaussian_copula_simulation(
    stock_returns: pd.DataFrame,
    weights: Optional[Union[pd.Series, np.ndarray]] = None,
    n_paths: int = 1000,
    horizon: int = TRADING_DAYS_PER_YEAR,
    seed: int = 42,
) -> Union[pd.DataFrame, np.ndarray]:
    """Joint-Gaussian Monte-Carlo over a return panel.

    Estimates per-stock mean and standard deviation along with a
    covariance matrix built from a robust correlation estimate
    (covariance is reconstructed as ``D Sigma D``).  When ``weights``
    is supplied, returns a ``(horizon + 1) x n_paths`` DataFrame of NAV
    paths starting from 1.0; otherwise returns a 3-D ndarray of shape
    ``(horizon, n_paths, n_stocks)`` of simulated daily returns.

    The covariance matrix is regularized by a tiny diagonal jitter and
    re-tried if the Cholesky decomposition fails, so the simulator is
    robust to nearly-singular correlation matrices.
    """
    if not isinstance(stock_returns, pd.DataFrame):
        raise TypeError("stock_returns must be a DataFrame")
    panel = stock_returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    panel = panel.fillna(0.0)
    if panel.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(int(seed))
    mu = panel.mean(axis=0).values
    sd = panel.std(axis=0, ddof=1).values
    sd = np.where(sd > 0, sd, 1e-8)
    corr = panel.corr().values
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    cov = (sd[:, None] * sd[None, :]) * corr

    # Cholesky with progressive jitter for nearly-singular covariance.
    L = None
    for jitter in (0.0, 1e-10, 1e-8, 1e-6, 1e-4):
        try:
            L = np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
            break
        except np.linalg.LinAlgError:
            continue
    if L is None:
        # Last-resort eigendecomposition fallback.
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-10, None)
        L = eigvecs * np.sqrt(eigvals)

    n_stocks = len(mu)
    z = rng.standard_normal(size=(int(horizon), int(n_paths), n_stocks))
    sim = mu + z @ L.T  # broadcast: (h, p, n)

    if weights is None:
        return sim

    w = pd.Series(weights, dtype="float64").reindex(panel.columns).fillna(0.0).values
    port_ret = sim @ w  # shape (horizon, n_paths)
    growth = 1.0 + port_ret
    nav = np.cumprod(growth, axis=0)
    nav = np.vstack([np.ones((1, int(n_paths))), nav])
    cols = [f"path_{i:04d}" for i in range(int(n_paths))]
    return pd.DataFrame(nav, columns=cols)
