"""Pairs trading rules built from standardized spread signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pairs_positions_from_zscore(
    zscore: pd.Series,
    asset_a: str,
    asset_b: str,
    entry_threshold: float = 1.5,
    exit_threshold: float = 0.5,
) -> pd.DataFrame:
    """Go long the spread when it is cheap and short it when it is rich.

    The rule has three triggers and one hold state:

    * ``z > entry``  -> asset_a is expensive relative to asset_b, so short the spread;
    * ``z < -entry`` -> asset_a is cheap, so go long the spread;
    * ``|z| < exit`` -> the spread is back near its mean, so close the trade;
    * anything else  -> keep whatever position is already open.

    Because the "hold" state simply carries the last decision forward, the whole rule is
    a forward fill of the three trigger states. Writing it that way instead of looping
    over dates gives identical positions and is dramatically faster, which matters when
    the walk-forward re-runs the rule once per fold.
    """

    # Mark only the dates where a trigger fires; leave the hold dates missing.
    signal = pd.Series(np.nan, index=zscore.index, dtype=float)
    signal[zscore > entry_threshold] = -1.0
    signal[zscore < -entry_threshold] = 1.0
    signal[zscore.abs() < exit_threshold] = 0.0
    # A missing z-score cannot trigger anything, so it must also be a hold.
    signal[zscore.isna()] = np.nan
    # Carry each decision forward until the next trigger; start flat.
    signal = signal.ffill().fillna(0.0)

    # The second leg takes the opposite side to form the pair trade.
    return pd.DataFrame({asset_a: signal, asset_b: -signal}, index=zscore.index)
