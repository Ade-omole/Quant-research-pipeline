"""Volatility-aware allocation rules.

These functions turn a volatility forecast into a position. The strategy logic
is simple: when forecast risk is high, hold less; when forecast risk is low,
hold more.
"""

from __future__ import annotations

import pandas as pd


def inverse_volatility_position(
    predicted_volatility: pd.Series,
    asset_name: str,
    max_leverage: float = 1.5,
) -> pd.DataFrame:
    """Allocate less capital when predicted volatility is high."""

    # Prevent division by zero if a volatility estimate is extremely small.
    safe_volatility = predicted_volatility.clip(lower=1e-6)
    # Inverse-volatility sizing means lower risk receives higher weight.
    raw_weight = 1.0 / safe_volatility
    # Normalize by the expanding average so weights stay around a stable scale.
    normalized_weight = raw_weight / raw_weight.expanding().mean() ## cumulative mean
    # Cap leverage so the rule cannot become unrealistically aggressive.
    clipped_weight = normalized_weight.clip(lower=0.0, upper=max_leverage)
    # Return a DataFrame because the backtest engine expects one column per asset.
    return clipped_weight.to_frame(name=asset_name)


def naive_buy_and_hold(asset_returns: pd.Series, asset_name: str) -> pd.DataFrame:
    """Create a constant long benchmark for comparison."""

    # A weight of 1.0 means fully invested in the asset every day.
    return pd.Series(1.0, index=asset_returns.index, name=asset_name).to_frame()
