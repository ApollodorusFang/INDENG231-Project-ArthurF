"""Load Nasdaq-100 daily close prices and compute returns.

The loader supports two CSV layouts so the pipeline can ingest the data
file regardless of how it was exported:

* **Wide format** -- the first column is a date, every other column is a
  ticker, and the cell values are close prices.
* **Long format** -- one row per (date, ticker) pair with at least the
  columns ``date``, ``ticker`` (or ``symbol``), and a price column such as
  ``close`` / ``adj_close`` / ``price``.

Tickers are read from the data, never hard-coded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]

_PRICE_COL_CANDIDATES = (
    "close",
    "adj_close",
    "adjclose",
    "adjusted_close",
    "adj close",
    "adjusted close",
    "price",
)


def _detect_long_format(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    return "date" in cols and ({"ticker", "symbol"} & cols) != set()


def load_price_data(path: PathLike) -> pd.DataFrame:
    """Load close-price data into a wide DataFrame indexed by date.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file containing daily close prices.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with a ``DatetimeIndex`` of trading days and one
        column per ticker. Missing prices are forward-filled, and any
        ticker that remains entirely NaN is dropped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Price data not found at {path}. "
            "Place 'nasdaq100_daily_5y.csv' under the 'data/' directory."
        )

    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    if _detect_long_format(df):
        date_col = cols_lower["date"]
        ticker_col = cols_lower.get("ticker") or cols_lower.get("symbol")
        price_col = next(
            (cols_lower[c] for c in _PRICE_COL_CANDIDATES if c in cols_lower),
            None,
        )
        if price_col is None:
            raise ValueError(
                "Long-format CSV must contain a price column "
                f"(one of {_PRICE_COL_CANDIDATES})."
            )
        df[date_col] = pd.to_datetime(df[date_col])
        prices = df.pivot_table(
            index=date_col, columns=ticker_col, values=price_col, aggfunc="last"
        )
    else:
        date_col = cols_lower.get("date", df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col])
        prices = df.set_index(date_col)

    prices.index = pd.DatetimeIndex(prices.index)
    prices.index.name = "date"
    prices = prices.sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.ffill().dropna(axis=1, how="all")
    prices.columns = [str(c) for c in prices.columns]
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from a wide price panel.

    The first row is filled with zeros so the returns DataFrame keeps the
    same index as ``prices``. NaN entries (e.g., a stock that has not
    started trading yet) are also replaced with zero so they contribute
    nothing to the portfolio return.
    """
    return prices.pct_change().fillna(0.0)
