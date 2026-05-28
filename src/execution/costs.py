"""Execution helpers that turn ideal positions into realistic returns.

Research signals often look better before trading frictions. This module adds
basic execution realism by measuring turnover, charging costs, and limiting how
quickly positions can change.
"""

from __future__ import annotations

import pandas as pd


def compute_turnover(positions: pd.DataFrame) -> pd.Series:
    """Turnover is the amount of position change from one period to the next."""

    # Difference positions through time, take absolute changes, then sum across assets.
    return positions.diff().abs().sum(axis=1).fillna(0.0).rename("turnover")


def compute_cost_rate(
    turnover: pd.Series,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> pd.Series:
    """Translate turnover into a return drag measured in decimal units."""

    # Combine explicit transaction costs and estimated slippage.
    total_cost_bps = transaction_cost_bps + slippage_bps
    # Higher turnover creates higher cost drag.
    cost_rate = turnover * (total_cost_bps / 10000.0)
    return cost_rate.rename("cost_rate")


def apply_turnover_cap(positions: pd.DataFrame, turnover_cap: float) -> pd.DataFrame:
    """Scale daily changes down when they exceed the allowed trading budget.

    This function preserves the direction of a desired position change but
    shrinks its size when the desired turnover is too large.
    """

    # Work on a copy so the original position table remains unchanged.
    capped_positions = positions.copy()

    # Start from the second date because the first date has no previous position.
    for timestamp in capped_positions.index[1:]:
        # Desired change is today's target position minus yesterday's position.
        desired_change = capped_positions.loc[timestamp] - capped_positions.shift(1).loc[timestamp]
        # Total trading required is the sum of absolute changes across columns.
        change_size = desired_change.abs().sum()

        # If desired turnover exceeds the cap, shrink the whole change vector.
        if change_size > turnover_cap and change_size > 0:
            # Scaling factor makes the final change exactly fit the turnover cap.
            scaling_factor = turnover_cap / change_size
            # Move only part of the way from yesterday's position to today's target.
            capped_positions.loc[timestamp] = (
                capped_positions.shift(1).loc[timestamp] + desired_change * scaling_factor
            )

    return capped_positions
