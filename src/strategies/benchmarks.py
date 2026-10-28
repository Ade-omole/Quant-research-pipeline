"""Benchmark position rules used for comparison: buy-and-hold and a monthly 60/40."""

from __future__ import annotations

import pandas as pd

from src.strategies.common import month_end_hold


def buy_and_hold_positions(index: pd.Index, asset: str) -> pd.DataFrame:
    """Fully invested in one asset on every date (weight 1.0)."""

    return pd.DataFrame(1.0, index=index, columns=[asset])


def sixty_forty_positions(index: pd.Index, equity: str, bond: str) -> pd.DataFrame:
    """60% equity and 40% bonds, rebalanced at every month-end."""

    target = pd.DataFrame({equity: 0.6, bond: 0.4}, index=index)
    return month_end_hold(target).fillna(0.0)
