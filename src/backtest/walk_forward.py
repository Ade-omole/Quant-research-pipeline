"""Expanding-window walk-forward estimation for strategy construction.

Each helper refits its model on an expanding historical slice, scores only the
following test slice, then concatenates disjoint out-of-sample test segments into
one position path. Regime switching uses train-period returns only when mapping
anonymous HMM states to bull/sideways/crisis labels.

Adaptive asset allocation has no model to refit: its weights are a deterministic
function of trailing prices. For it the walk-forward instead *selects parameters*.
On every fold each candidate parameter set is scored by its net Sharpe on the
training window only, and the winner's positions are used on the following test
block. That answers the question "what would this rule have earned if I had not
known the right lookback in advance?".

Walk-forward modules return **positions only**. The notebook (or any caller) should
feed those positions through ``positions_to_strategy_returns`` with your
``transaction_cost_bps`` and ``slippage_bps`` so net PnL matches the friction model.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.backtest.engine import positions_to_strategy_returns
from src.models.ou import compute_spread_zscore
from src.models.regime import fit_hmm_regime_model, infer_regimes
from src.models.volatility import fit_random_forest_volatility_model, predict_random_forest_volatility
from src.strategies.pairs_trading import pairs_positions_from_zscore
from src.strategies.regime_switching import build_regime_label_map, regime_based_positions
from src.strategies import adaptive_allocation
from src.strategies.volatility_timing import inverse_volatility_position
from src.utils.config import (
    AAA_CONFIG, DATA_CONFIG, PAIRS_CONFIG, WALK_FORWARD_CONFIG, PairsConfig, WalkForwardConfig,
)
from src.utils.data_loader import estimate_pair_ols_coefficients, spread_from_prices


def infer_walk_forward_calendar(
    volatility_features: pd.DataFrame,
    regime_features: pd.DataFrame,
    prices_a: pd.Series | None = None,
    prices_b: pd.Series | None = None,
    *,
    extra_indexes: list[pd.Index] | None = None,
) -> pd.DatetimeIndex:
    """Trading dates where volatility features, regime features, and pair prices are all valid.

    Walk-forward uses integer split boundaries; if each strategy supplies a different row
    calendar, their OOS test dates almost never overlap and a joint ``DataFrame`` with
    ``dropna(how='any')`` ends up empty — ``performance_metrics`` then returns all zeros.
    Passing this shared index into every walker keeps train/test windows aligned.
    """

    vol_ok = volatility_features.dropna(how="any").index
    reg_ok = regime_features.dropna(how="any").index
    # The pair legs are optional. Since the pairs sleeve now scans every pair rather
    # than trading one configured pair, the shared calendar should not be narrowed by
    # whichever pair happened to be the default; each pair contributes zero positions
    # on dates where one of its own legs has not listed.
    if prices_a is not None and prices_b is not None:
        pair_ok = pd.concat([prices_a, prices_b], axis=1).dropna(how="any").index
    else:
        pair_ok = vol_ok
    # Shared calendar keeps every walk-forward path on the same train/test dates.
    common = vol_ok.intersection(reg_ok).intersection(pair_ok)
    # Rule-based strategies (adaptive allocation) contribute the dates on which their
    # signal exists, so they share the folds rather than defining their own.
    for extra in extra_indexes or []:
        common = common.intersection(pd.Index(extra))
    return pd.DatetimeIndex(common.sort_values())


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


# ---------------------------------------------------------------------------
# Walk-forward parameter selection for the rule-based strategy
# ---------------------------------------------------------------------------


def training_sharpe(
    returns: pd.Series,
    annualization_factor: int = 252,
    min_observations: int = 63,
) -> float:
    """Annualized Sharpe used to score candidate parameter sets on a training window."""

    clean = returns.dropna()
    if len(clean) < min_observations or clean.std() == 0:
        return float("-inf")
    return float(np.sqrt(annualization_factor) * clean.mean() / clean.std())


def adaptive_allocation_candidates(
    prices: pd.DataFrame,
    cfg: WalkForwardConfig = WALK_FORWARD_CONFIG,
) -> dict[str, pd.DataFrame]:
    """Position tables for every (lookback, top_n) pair in the walk-forward grid.

    The rule is causal: the weight on day t depends only on prices up to day t, so a
    table built on the full history is identical, row by row, to one built on any
    history truncated at t. Building the candidates once is therefore both correct
    and fast; the walk-forward then only decides which candidate is live.
    ``tests/test_walk_forward_integrity.py`` verifies this property directly.
    """

    return {
        f"lb{lookback}_top{top_n}": adaptive_allocation.build_positions(
            prices, replace(AAA_CONFIG, lookback=lookback, top_n=top_n)
        )
        for lookback in cfg.aaa_lookbacks
        for top_n in cfg.aaa_top_n
    }


def walk_forward_rule_positions(
    candidate_positions: dict[str, pd.DataFrame],
    asset_returns: pd.DataFrame,
    *,
    calendar: pd.DatetimeIndex | pd.Index,
    min_train_observations: int | None = None,
    test_observations: int | None = None,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Expanding-window parameter selection by training-window net Sharpe.

    Returns the stitched out-of-sample positions and a Series recording which
    candidate was chosen at the start of each test block.
    """

    idx = pd.DatetimeIndex(pd.Index(calendar).drop_duplicates().sort_values())
    if min_train_observations is None:
        min_train_observations = WALK_FORWARD_CONFIG.min_train_observations
    if test_observations is None:
        test_observations = WALK_FORWARD_CONFIG.test_observations

    splits = expanding_walk_forward_splits(idx, min_train_observations, test_observations)
    all_columns = sorted({column for positions in candidate_positions.values() for column in positions.columns})
    if not splits:
        return pd.DataFrame(columns=all_columns), pd.Series(dtype=object)

    # Net return of every candidate over the whole history. Each candidate pays its
    # own turnover cost, so a parameter set that trades more is penalised for it.
    candidate_net = {
        label: positions_to_strategy_returns(
            positions, asset_returns, transaction_cost_bps, slippage_bps
        )["net_return"]
        for label, positions in candidate_positions.items()
    }

    chunks: list[pd.DataFrame] = []
    choices: dict[pd.Timestamp, str] = {}
    for train_index, test_index in splits:
        # Score every candidate on the training window only.
        scores = {label: training_sharpe(net.reindex(train_index)) for label, net in candidate_net.items()}
        best_label = max(scores, key=scores.get)
        # Trade the winner on the following test block only.
        chunk = candidate_positions[best_label].reindex(index=test_index, columns=all_columns).fillna(0.0)
        chunks.append(chunk)
        choices[test_index[0]] = best_label

    stitched = pd.concat(chunks).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    return stitched, pd.Series(choices, name="selected_parameters")


def pair_candidates(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex | pd.Index,
    *,
    cfg: PairsConfig = PAIRS_CONFIG,
    min_train_observations: int | None = None,
    test_observations: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Walk-forward position tables for every eligible pair.

    Each pair's hedge ratio is refit on the training prefix of every fold, exactly as
    ``walk_forward_pairs_positions`` already does, so these tables are themselves
    out of sample. Selecting among them on training performance therefore adds a
    second, outer layer of walk-forward rather than reintroducing hindsight.
    """

    import itertools

    pairs = cfg.candidate_pairs or list(itertools.combinations(sorted(DATA_CONFIG.symbols), 2))
    return {
        f"{asset_a}-{asset_b}": walk_forward_pairs_positions(
            prices[asset_a], prices[asset_b], asset_a, asset_b,
            calendar=calendar, zscore_window=cfg.zscore_window,
            min_train_observations=min_train_observations,
            test_observations=test_observations,
        )
        for asset_a, asset_b in pairs
    }


def walk_forward_pair_selection(
    prices: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    calendar: pd.DatetimeIndex | pd.Index,
    cfg: PairsConfig = PAIRS_CONFIG,
    min_train_observations: int | None = None,
    test_observations: int | None = None,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Choose which pair to trade on each fold, using training data only.

    Hard-coding one pair is a hindsight decision: the pair that looks best over the
    whole sample was chosen knowing the whole sample. Here the eligible pairs are
    scored by net Sharpe ratio on each expanding training window, and the winner is
    traded on the following test block. Returns the stitched positions and a record
    of which pair was live when.
    """

    candidates = pair_candidates(
        prices, calendar, cfg=cfg,
        min_train_observations=min_train_observations,
        test_observations=test_observations,
    )
    return walk_forward_rule_positions(
        candidates, asset_returns, calendar=calendar,
        min_train_observations=min_train_observations,
        test_observations=test_observations,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
