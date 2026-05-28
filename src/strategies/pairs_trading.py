"""Pairs trading rules built from standardized spread signals."""

from __future__ import annotations

import pandas as pd


def pairs_positions_from_zscore(
    zscore: pd.Series,
    asset_a: str,
    asset_b: str,
    entry_threshold: float = 1.5,
    exit_threshold: float = 0.5,
) -> pd.DataFrame:
    """Go long the spread when it is cheap and short it when it is rich."""

    # Start with zero exposure on every date for both assets.
    positions = pd.DataFrame(0.0, index=zscore.index, columns=[asset_a, asset_b])
    # Track the current signal so the strategy can hold a trade until exit.
    current_signal = 0.0

    # Iterate through valid z-score observations in chronological order.
    for timestamp, value in zscore.dropna().items():
        # A high z-score means asset_a is expensive relative to asset_b.
        # Short the spread: short asset_a and long asset_b.
        if value > entry_threshold:
            current_signal = -1.0
        # A low z-score means asset_a is cheap relative to asset_b.
        # Long the spread: long asset_a and short asset_b.
        elif value < -entry_threshold:
            current_signal = 1.0
        # Once the spread comes close to normal, close the trade.
        elif abs(value) < exit_threshold:
            current_signal = 0.0

        # The first asset receives the signal direction.
        positions.loc[timestamp, asset_a] = current_signal
        # The second asset receives the opposite direction to form a pair trade.
        positions.loc[timestamp, asset_b] = -current_signal # or -current_signal * hedge_ratio (better)

    return positions
