"""Regime-aware allocation rules.

The HMM produces numeric states such as 0, 1, and 2. This module interprets
those states economically and converts them into portfolio exposure.
"""

from __future__ import annotations

import pandas as pd


def build_regime_label_map(regimes: pd.Series, returns: pd.Series) -> dict[int, str]:
    """Map HMM state ids to bull/sideways/crisis from average return on this slice."""

    regime_table = pd.DataFrame({"regime": regimes, "returns": returns.reindex(regimes.index)})
    regime_order = regime_table.groupby("regime")["returns"].mean().sort_values()
    label_map: dict[int, str] = {}
    # With three states, interpret the lowest-return state as crisis and the highest as bull.
    if len(regime_order) >= 3:
        label_map[int(regime_order.index[0])] = "crisis"
        label_map[int(regime_order.index[1])] = "sideways"
        label_map[int(regime_order.index[-1])] = "bull"
    else:
        # If fewer states exist, avoid over-interpreting the model.
        for regime_value in regime_order.index:
            label_map[int(regime_value)] = "sideways"
    return label_map


def map_regimes_to_labels(regimes: pd.Series, returns: pd.Series) -> pd.DataFrame:
    """Interpret raw HMM states using each state's average return."""

    # Combine state labels and realized returns into one table for analysis.
    regime_table = pd.DataFrame({"regime": regimes, "returns": returns.reindex(regimes.index)})
    # Sort regimes by average return from worst to best.
    label_map = build_regime_label_map(regimes, returns)
    regime_table["regime_label"] = regime_table["regime"].map(label_map)
    return regime_table


def regime_based_positions(
    regime_labels: pd.Series,
    target_asset: str,
) -> pd.DataFrame:
    """Use simple state-dependent exposure rules."""

    # Exposure is highest in bull regimes and lowest in crisis regimes.
    weight_map = {
        "bull": 1.0,
        "sideways": 0.35,
        "crisis": 0.0,
    }
    # Convert labels into numeric portfolio weights.
    weights = regime_labels.map(weight_map).fillna(0.0)
    # Return a DataFrame because the backtest engine expects one column per asset.
    return weights.to_frame(name=target_asset)
