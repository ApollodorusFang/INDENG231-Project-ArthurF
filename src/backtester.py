"""Daily-close backtesting engine.

The engine iterates over the trading days in a price panel.  At each day
``t`` it:

1. Slices the prices/returns to the rows up to and including ``t``.
2. Asks the strategy for target weights using *only* that history.
3. Validates the weights (no shorts, no leverage, full alignment).
4. Computes turnover relative to the previous weights and applies an
   optional proportional transaction cost.
5. Realizes those weights against the **next** day's returns to update
   the portfolio NAV.

That last point is the structural reason the backtester is free of
look-ahead bias: the strategy never sees ``r_{t+1}`` when choosing
``w_t``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_TRANSACTION_COST, INITIAL_CAPITAL
from .data_loader import compute_returns


@dataclass
class BacktestResult:
    """Container for the standard outputs of a backtest run."""

    nav: pd.Series
    daily_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    log: List[dict] = field(default_factory=list)
    strategy_name: str = ""
    initial_capital: float = INITIAL_CAPITAL
    transaction_cost: float = DEFAULT_TRANSACTION_COST


class Backtester:
    """Run a strategy against a pre-loaded daily price panel."""

    def __init__(
        self,
        prices: pd.DataFrame,
        returns: Optional[pd.DataFrame] = None,
        transaction_cost: float = DEFAULT_TRANSACTION_COST,
        initial_capital: float = INITIAL_CAPITAL,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
    ) -> None:
        prices = prices.sort_index()
        if returns is None:
            returns = compute_returns(prices)
        if start is not None:
            prices = prices.loc[start:]
            returns = returns.loc[start:]
        if end is not None:
            prices = prices.loc[:end]
            returns = returns.loc[:end]

        self.prices: pd.DataFrame = prices
        self.returns: pd.DataFrame = returns
        self.universe: list[str] = list(prices.columns)
        self.transaction_cost: float = float(transaction_cost)
        self.initial_capital: float = float(initial_capital)

    def run(self, strategy: Any) -> BacktestResult:
        """Execute ``strategy`` over the configured date range."""
        dates = list(self.prices.index)
        n = len(dates)
        if n == 0:
            raise ValueError("Backtester received an empty price panel.")

        weights_history = pd.DataFrame(0.0, index=dates, columns=self.universe)
        turnover = pd.Series(0.0, index=dates)
        daily_ret = pd.Series(0.0, index=dates)
        nav_series = pd.Series(index=dates, dtype=float)
        nav_series.iloc[0] = self.initial_capital
        log: list[dict] = []

        prev_w = pd.Series(0.0, index=self.universe)
        nav = self.initial_capital

        for i, t in enumerate(dates):
            # Information set: closes up to and including day t -- nothing later.
            prices_t = self.prices.iloc[: i + 1]
            returns_t = self.returns.iloc[: i + 1]

            try:
                raw = strategy.generate_weights(prices_t, returns_t, t)
            except Exception as exc:  # noqa: BLE001
                log.append(
                    {"date": str(t.date()), "event": "strategy_error", "error": repr(exc)}
                )
                raw = pd.Series(0.0, index=self.universe)

            w = strategy.validate_weights(raw, self.universe)
            weights_history.loc[t] = w.values

            tov = float((w - prev_w).abs().sum())
            turnover.iloc[i] = tov
            cost = self.transaction_cost * tov

            # Apply target weights to NEXT-day returns -- this is the
            # mechanism that prevents look-ahead bias: w_t is chosen using
            # information through t, but is only realized against r_{t+1}.
            if i + 1 < n:
                next_ret = (
                    self.returns.iloc[i + 1]
                    .reindex(self.universe)
                    .fillna(0.0)
                )
                port_ret = float((w.values * next_ret.values).sum()) - cost
                nav = nav * (1.0 + port_ret)
                nav_series.iloc[i + 1] = nav
                daily_ret.iloc[i + 1] = port_ret

                # Optional reward-feedback hook for adaptive strategies
                # (e.g. the UCB meta-strategy).  Called only after the
                # next-day return is realized, so the strategy never
                # peeks at r_{t+1} when picking w_t.
                if hasattr(strategy, "update_after_return"):
                    try:
                        strategy.update_after_return(port_ret, t)
                    except Exception as exc:  # noqa: BLE001
                        log.append(
                            {
                                "date": str(t.date()),
                                "event": "update_after_return_error",
                                "error": repr(exc),
                            }
                        )
            else:
                # On the final day there is no t+1 return to realize.
                # The decision still incurs any transaction cost on the
                # rebalance, which we charge here.
                if cost != 0.0:
                    nav = nav * (1.0 - cost)
                    nav_series.iloc[i] = nav
                    daily_ret.iloc[i] = -cost

            log.append(
                {
                    "date": str(t.date()),
                    "n_holdings": int((w > 0).sum()),
                    "weight_sum": float(w.sum()),
                    "cash_weight": float(max(0.0, 1.0 - w.sum())),
                    "turnover": tov,
                    "cost": cost,
                }
            )
            prev_w = w

        nav_series = nav_series.ffill()

        return BacktestResult(
            nav=nav_series,
            daily_returns=daily_ret,
            weights=weights_history,
            turnover=turnover,
            log=log,
            strategy_name=getattr(strategy, "name", strategy.__class__.__name__),
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
        )
