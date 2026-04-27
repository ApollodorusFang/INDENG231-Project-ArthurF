"""Project-wide configuration: paths and core constants.

All paths are derived from ``PROJECT_ROOT`` so that modules and entry-point
scripts can be executed from the repository root without further setup.
Output directories are created on import to keep callers concise.
"""
from __future__ import annotations

from pathlib import Path

# Repository root = the parent of the ``src`` directory that contains this file.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Input data
DATA_PATH: Path = PROJECT_ROOT / "data" / "nasdaq100_daily_5y.csv"

# Output locations
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
FIGURE_DIR: Path = OUTPUT_DIR / "figures"
TABLE_DIR: Path = OUTPUT_DIR / "tables"
LOG_DIR: Path = OUTPUT_DIR / "logs"

# Backtest constants
TRADING_DAYS_PER_YEAR: int = 252
INITIAL_CAPITAL: float = 1.0
DEFAULT_TRANSACTION_COST: float = 0.0

for _d in (OUTPUT_DIR, FIGURE_DIR, TABLE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
