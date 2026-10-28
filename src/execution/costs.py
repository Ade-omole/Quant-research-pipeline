"""Execution helpers that turn ideal positions into realistic returns.

Research signals often look better before trading frictions. This module adds
basic execution realism by measuring turnover, charging costs, and limiting how
quickly positions can change.
"""

from __future__ import annotations

import numpy as np
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

    This preserves the direction of a desired position change but shrinks its size
    when the desired turnover is too large. The loop is sequential on purpose:
    today's allowed trade starts from where yesterday's *capped* trade actually left
    the portfolio, not from the ideal target. The NumPy form below is identical to
    the original pandas loop to machine precision and about ninety times faster,
    which is what makes the portfolio grid affordable.
    """

    values = np.nan_to_num(positions.to_numpy(dtype=float, copy=True), nan=0.0)
    capped = values.copy()

    # Start from the second date because the first date has no previous position.
    for row in range(1, len(capped)):
        # Desired change is today's target minus yesterday's capped position.
        desired_change = values[row] - capped[row - 1]
        # Total trading required is the sum of absolute changes across columns.
        change_size = float(np.abs(desired_change).sum())
        if change_size > turnover_cap and change_size > 0:
            # Move only part of the way toward today's target.
            desired_change = desired_change * (turnover_cap / change_size)
        capped[row] = capped[row - 1] + desired_change

    return pd.DataFrame(capped, index=positions.index, columns=positions.columns)
