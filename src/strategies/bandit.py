"""UCB meta-strategy that picks among existing portfolio strategies.

Each base strategy is treated as one bandit *arm*.  At every trading
day the meta-strategy selects an arm via the standard UCB-1 rule

    UCB_k = mean_reward_k + c * sqrt(log(total_trials) / n_k)

asks that arm for its target weights, and forwards the result.  After
the day's return is realized the backtester calls back into
:py:meth:`UCBMetaStrategy.update_after_return`, which credits the
chosen arm with the realized portfolio return.  Because the reward
feedback only arrives *after* the realized return, the selection
process never peeks at future data.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .base import BaseStrategy

ArmContainer = Union[Sequence[BaseStrategy], Mapping[str, BaseStrategy]]


class UCBMetaStrategy(BaseStrategy):
    """Multi-armed-bandit selector over a fixed set of base strategies."""

    def __init__(self, arms: ArmContainer, c: float = 1.0) -> None:
        if isinstance(arms, Mapping):
            arm_names = list(arms.keys())
            arm_list = [arms[k] for k in arm_names]
        else:
            arm_list = list(arms)
            arm_names = [
                getattr(a, "name", f"arm_{i}") for i, a in enumerate(arm_list)
            ]

        if len(arm_list) == 0:
            raise ValueError("UCBMetaStrategy needs at least one arm.")

        self.arms: List[BaseStrategy] = arm_list
        self.arm_names: List[str] = arm_names
        self.c: float = float(c)
        self.n_arms: int = len(arm_list)

        self.counts: List[int] = [0] * self.n_arms
        self.cumulative_reward: List[float] = [0.0] * self.n_arms
        self.total_trials: int = 0

        # History of (decision_date, arm_idx, realized_reward).
        self.history: List[Tuple[pd.Timestamp, int, float]] = []

        # Pending bookkeeping: which arm answered the most recent
        # generate_weights call and on what date.  Cleared by
        # update_after_return so each decision is credited exactly once.
        self._pending_arm: Optional[int] = None
        self._pending_date: Optional[pd.Timestamp] = None

        self.name = f"ucb_meta_c{self.c:g}_n{self.n_arms}"

    # -- arm selection ----------------------------------------------------

    def _select_arm(self) -> int:
        """Pick an arm: untried first, else max-UCB."""
        for i, n in enumerate(self.counts):
            if n == 0:
                return i
        log_total = np.log(max(self.total_trials, 1))
        ucb_scores = np.array(
            [
                (self.cumulative_reward[i] / self.counts[i])
                + self.c * np.sqrt(log_total / self.counts[i])
                for i in range(self.n_arms)
            ]
        )
        return int(np.argmax(ucb_scores))

    # -- BaseStrategy contract -------------------------------------------

    def generate_weights(
        self,
        prices_until_t: pd.DataFrame,
        returns_until_t: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> pd.Series:
        idx = self._select_arm()
        self._pending_arm = idx
        self._pending_date = current_date
        arm = self.arms[idx]
        return arm.generate_weights(prices_until_t, returns_until_t, current_date)

    # -- reward feedback hook (called by Backtester) ---------------------

    def update_after_return(
        self,
        realized_return: float,
        current_date: pd.Timestamp,
    ) -> None:
        """Credit the most recently selected arm with ``realized_return``."""
        if self._pending_arm is None:
            return
        i = self._pending_arm
        decision_date = self._pending_date if self._pending_date is not None else current_date

        self.counts[i] += 1
        self.cumulative_reward[i] += float(realized_return)
        self.total_trials += 1
        self.history.append((decision_date, i, float(realized_return)))

        self._pending_arm = None
        self._pending_date = None

    # -- introspection helpers -------------------------------------------

    def selection_counts(self) -> Dict[str, int]:
        """Return a name -> selection-count mapping."""
        return dict(zip(self.arm_names, self.counts))

    def mean_rewards(self) -> Dict[str, float]:
        """Return a name -> realized-mean-reward mapping (0 for untried arms)."""
        out: Dict[str, float] = {}
        for name, n, r in zip(self.arm_names, self.counts, self.cumulative_reward):
            out[name] = (r / n) if n > 0 else 0.0
        return out

    def selection_history_df(self) -> pd.DataFrame:
        """Per-decision history as a DataFrame."""
        if not self.history:
            return pd.DataFrame(columns=["date", "arm_idx", "arm", "reward"])
        rows = [
            {"date": d, "arm_idx": i, "arm": self.arm_names[i], "reward": r}
            for d, i, r in self.history
        ]
        return pd.DataFrame(rows).set_index("date")
