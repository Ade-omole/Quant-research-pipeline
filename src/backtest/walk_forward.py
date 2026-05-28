"""Expanding-window walk-forward estimation for strategy construction.

Each helper refits its model on an expanding historical slice, scores only the
following test slice, then concatenates disjoint out-of-sample test segments into
one position path. Regime switching uses train-period returns only when mapping
anonymous HMM states to bull/sideways/crisis labels.

Walk-forward modules return **positions only**. The notebook (or any caller) should
feed those positions through ``positions_to_strategy_returns`` with your
``transaction_cost_bps`` and ``slippage_bps`` so net PnL matches the friction model.
"""

from __future__ import annotations

import pandas as pd

from src.models.ou import compute_spread_zscore
from src.models.regime import fit_hmm_regime_model, infer_regimes
from src.models.volatility import fit_random_forest_volatility_model, predict_random_forest_volatility
from src.strategies.pairs_trading import pairs_positions_from_zscore
from src.strategies.regime_switching import build_regime_label_map, regime_based_positions
from src.strategies.volatility_timing import inverse_volatility_position
from src.utils.data_loader import estimate_pair_ols_coefficients, spread_from_prices


def infer_walk_forward_calendar(
    volatility_features: pd.DataFrame,
    regime_features: pd.DataFrame,
    prices_a: pd.Series,
    prices_b: pd.Series,
) -> pd.DatetimeIndex:
    """Trading dates where volatility features, regime features, and pair prices are all valid.

    Walk-forward uses integer split boundaries; if each strategy supplies a different row
    calendar, their OOS test dates almost never overlap and a joint ``DataFrame`` with
    ``dropna(how='any')`` ends up empty — ``performance_metrics`` then returns all zeros.
    Passing this shared index into every walker keeps train/test windows aligned.
    """

    vol_ok = volatility_features.dropna(how="any").index
    reg_ok = regime_features.dropna(how="any").index
    pair_ok = pd.concat([prices_a, prices_b], axis=1).dropna(how="any").index
    # Shared calendar keeps all three walk-forward paths on the same train/test dates.
    common = vol_ok.intersection(reg_ok).intersection(pair_ok).sort_values()
    return pd.DatetimeIndex(common)


def expanding_walk_forward_splits(
    index: pd.Index,
    min_train_observations: int,
    test_observations: int,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Non-overlapping test blocks with an expanding training prefix.

    Fold k uses ``train = sorted_index[:boundary]`` and a test block of length
    ``test_observations`` immediately following, then advances ``boundary`` by the
    test length so training history grows across folds.
    """

    idx = pd.DatetimeIndex(pd.Index(index).drop_duplicates().sort_values())
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    boundary = min_train_observations
    while boundary + test_observations <= len(idx):
        train = idx[:boundary]
        test = idx[boundary : boundary + test_observations]
        splits.append((train, test))
        boundary += test_observations
    return splits


def walk_forward_volatility_positions(
    volatility_features: pd.DataFrame,
    benchmark_asset: str,
    min_train_observations: int | None = None,
    test_observations: int | None = None,
    *,
    calendar: pd.DatetimeIndex | pd.Index | None = None,
) -> pd.DataFrame:
    """Random forest refit on expanding history; positions from OOS forecasts only."""

    if calendar is not None:
        feature_index = pd.DatetimeIndex(pd.Index(calendar).drop_duplicates().sort_values())
    else:
        feature_index = volatility_features.index.drop_duplicates().sort_values()
    if min_train_observations is None:
        min_train_observations = max(252 * 2, len(feature_index) // 5)
    if test_observations is None:
        test_observations = 21

    splits = expanding_walk_forward_splits(feature_index, min_train_observations, test_observations)
    if not splits:
        return pd.DataFrame(columns=[benchmark_asset])

    forecast_chunks: list[pd.Series] = []
    min_rf_rows = 50
    for train_index, test_index in splits:
        train_frame = volatility_features.reindex(train_index).dropna(how="any")
        test_frame = volatility_features.reindex(test_index).dropna(how="any")
        if len(train_frame) < min_rf_rows or test_frame.empty:
            continue
        model = fit_random_forest_volatility_model(train_frame)
        forecasts = predict_random_forest_volatility(model, test_frame)
        forecast_chunks.append(forecasts)

    if not forecast_chunks:
        return pd.DataFrame(columns=[benchmark_asset])

    combined = pd.concat(forecast_chunks).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return inverse_volatility_position(combined, benchmark_asset)


def walk_forward_regime_positions(
    regime_features: pd.DataFrame,
    benchmark_returns: pd.Series,
    benchmark_asset: str,
    *,
    min_train_observations: int | None = None,
    test_observations: int | None = None,
    calendar: pd.DatetimeIndex | pd.Index | None = None,
) -> pd.DataFrame:
    """HMM refit on expanding history; test states labeled using train returns only."""

    if calendar is not None:
        feature_index = pd.DatetimeIndex(pd.Index(calendar).drop_duplicates().sort_values())
    else:
        feature_index = regime_features.index.drop_duplicates().sort_values()
    if min_train_observations is None:
        min_train_observations = max(252 * 2, len(feature_index) // 5)
    if test_observations is None:
        test_observations = 21

    splits = expanding_walk_forward_splits(feature_index, min_train_observations, test_observations)
    if not splits:
        return pd.DataFrame(columns=[benchmark_asset])

    min_hmm_rows = 100
    weight_chunks: list[pd.Series] = []
    for train_index, test_index in splits:
        train_feat = regime_features.reindex(train_index).dropna(how="any")
        test_feat = regime_features.reindex(test_index).dropna(how="any")
        if len(train_feat) < min_hmm_rows or test_feat.empty:
            continue
        try:
            model = fit_hmm_regime_model(
                train_feat,
                covariance_type="diag",
                min_covar=1e-2,
            )
            regimes_train = infer_regimes(model, train_feat)
            regimes_test = infer_regimes(model, test_feat)
            returns_train = benchmark_returns.reindex(regimes_train.index)
            label_map = build_regime_label_map(regimes_train, returns_train)
            labels_test = regimes_test.map(label_map).fillna("sideways")
            positions = regime_based_positions(labels_test, benchmark_asset)
            weight_chunks.append(positions[benchmark_asset])
        except Exception:
            # hmmlearn can raise ValueError (non-PSD covars) or linalg errors in decode.
            continue

    if not weight_chunks:
        return pd.DataFrame(columns=[benchmark_asset])

    combined = pd.concat(weight_chunks).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.to_frame(name=benchmark_asset)


def walk_forward_pairs_positions(
    prices_a: pd.Series,
    prices_b: pd.Series,
    asset_a: str,
    asset_b: str,
    *,
    min_train_observations: int | None = None,
    test_observations: int | None = None,
    zscore_window: int = 21,
    calendar: pd.DatetimeIndex | pd.Index | None = None,
) -> pd.DataFrame:
    """Rolling OLS hedge on expanding history; pair rules scored on OOS test windows only."""

    aligned = pd.concat([prices_a, prices_b], axis=1).dropna(how="any")
    idx = pd.DatetimeIndex(pd.Index(calendar).drop_duplicates().sort_values()) if calendar is not None else aligned.index
    if min_train_observations is None:
        min_train_observations = max(252 * 2, len(idx) // 5)
    if test_observations is None:
        test_observations = 21

    splits = expanding_walk_forward_splits(idx, min_train_observations, test_observations)
    if not splits:
        return pd.DataFrame(columns=[asset_a, asset_b])

    position_chunks: list[pd.DataFrame] = []
    min_pair_train = 50 # minimum number of observations to estimate the pair hedge ratio
    for train_index, test_index in splits:
        train_panel = aligned.reindex(train_index).dropna(how="any")
        if len(train_panel) < min_pair_train:
            continue # skip if not enough observations to estimate the pair hedge ratio
        train_a = train_panel.iloc[:, 0]
        train_b = train_panel.iloc[:, 1]
        intercept, hedge = estimate_pair_ols_coefficients(train_a, train_b)
        combined_index = train_index.union(test_index).sort_values()
        spread_path = spread_from_prices(
            prices_a.reindex(combined_index),
            prices_b.reindex(combined_index),
            intercept,
            hedge,
        )
        zscores = compute_spread_zscore(spread_path, window=zscore_window)
        positions_full = pairs_positions_from_zscore(zscores, asset_a, asset_b)
        position_chunks.append(positions_full.reindex(test_index).fillna(0.0)) # keep only the test index and fill nan with 0

    if not position_chunks:
        return pd.DataFrame(columns=[asset_a, asset_b])

    out = pd.concat(position_chunks).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out
