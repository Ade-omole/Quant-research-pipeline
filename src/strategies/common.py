"""Helpers shared by the rule-based strategies.

Everything here is written to tolerate a *ragged* price panel: in the early years
of the sample three of the eight ETFs did not exist yet, so their columns are
missing rather than zero. Missing assets must be excluded from rankings and from
weight normalization, never treated as a zero-return holding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import DATA_CONFIG


def annualized_rolling_volatility(
    returns: pd.DataFrame,
    window: int,
    min_vol: float = 0.05,
) -> pd.DataFrame:
    """Trailing annualized volatility with a floor so that 1/vol stays bounded.

    Assets with fewer than ``window`` observations keep a missing value, which is
    what stops them from receiving a weight before they exist.
    """

    volatility = returns.rolling(window).std() * np.sqrt(DATA_CONFIG.annualization_factor)
    return volatility.clip(lower=min_vol)


def month_end_hold(weights: pd.DataFrame) -> pd.DataFrame:
    """Keep only month-end rows, then hold those weights until the next month-end.

    This turns a daily signal into a monthly rebalanced position, so turnover (and
    therefore cost) is incurred once a month rather than every day.
    """

    month_ends = weights.groupby(pd.Grouper(freq="ME")).tail(1).index
    return weights.loc[weights.index.isin(month_ends)].reindex(weights.index).ffill()


def first_date_with_min_assets(frame: pd.DataFrame, min_assets: int) -> pd.Timestamp:
    """First date on which at least ``min_assets`` columns hold a valid value."""

    usable = frame.notna().sum(axis=1) >= min_assets
    if not usable.any():
        raise ValueError(f"No date has {min_assets} valid assets.")
    return usable.idxmax()
