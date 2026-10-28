"""Robustness checks: sub-periods, cost sensitivity, parameter grids, regime conditioning."""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd

from src.backtest.engine import performance_metrics, positions_to_strategy_returns
from src.utils.config import DATA_CONFIG, ROBUSTNESS_CONFIG

ANN = DATA_CONFIG.annualization_factor


def sharpe(returns: pd.Series, min_observations: int = 100) -> float:
    """Annualized Sharpe; NaN when the sample is too short to be meaningful."""

    clean = returns.dropna()
    if len(clean) < min_observations or clean.std() == 0:
        return float("nan")
    return float(np.sqrt(ANN) * clean.mean() / clean.std())


def subperiod_sharpes(
    named_returns: dict[str, pd.Series],
    subperiods: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Net Sharpe of every return stream inside every sub-period (rows = names)."""

    subperiods = subperiods or ROBUSTNESS_CONFIG.subperiods
    return pd.DataFrame({
        name: {label: sharpe(returns.loc[start:end]) for label, (start, end) in subperiods.items()}
        for name, returns in named_returns.items()
    }).T


def cost_sensitivity(
    positions_by_name: dict[str, pd.DataFrame],
    asset_returns: pd.DataFrame,
    cost_grid_bps: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Net Sharpe of each strategy as the total cost per unit of turnover rises."""

    cost_grid_bps = cost_grid_bps or ROBUSTNESS_CONFIG.cost_grid_bps
    rows = {}
    for name, positions in positions_by_name.items():
        rows[name] = {
            f"{bps:g}bps": sharpe(
                positions_to_strategy_returns(positions, asset_returns, bps, 0.0)["net_return"]
            )
            for bps in cost_grid_bps
        }
    return pd.DataFrame(rows).T


def parameter_grid(
    prices: pd.DataFrame,
    asset_returns: pd.DataFrame,
    builder: Callable[[pd.DataFrame, object], pd.DataFrame],
    base_config: object,
    grid: dict[str, list],
    transaction_cost_bps: float,
    slippage_bps: float,
    metric: str = "sharpe_ratio",
    start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Evaluate a rule-based strategy over a 1-D or 2-D parameter grid.

    With one parameter the result has one column; with two parameters the first
    parameter indexes the rows and the second the columns. Pass ``start`` to score
    every cell on the same window as the rest of the project.
    """

    names = list(grid)
    if len(names) not in (1, 2):
        raise ValueError("parameter_grid supports one or two parameters")

    def evaluate(**params) -> float:
        cfg = replace(base_config, **params)
        positions = builder(prices, cfg)
        net = positions_to_strategy_returns(positions, asset_returns, transaction_cost_bps, slippage_bps)["net_return"]
        # Restricting to the out-of-sample window keeps every number in the project
        # on one common evaluation period.
        if start is not None:
            net = net.loc[start:]
        return performance_metrics(net)[metric]

    if len(names) == 1:
        name = names[0]
        return pd.DataFrame({metric: {f"{name}={v}": evaluate(**{name: v}) for v in grid[name]}})

    row_name, col_name = names
    table = {}
    for row_value in grid[row_name]:
        table[f"{row_name}={row_value}"] = {
            f"{col_name}={col_value}": evaluate(**{row_name: row_value, col_name: col_value})
            for col_value in grid[col_name]
        }
    return pd.DataFrame(table).T


def regime_conditioned_table(
    named_returns: dict[str, pd.Series],
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """Annualized return, volatility and Sharpe of each stream inside each regime label."""

    rows = {}
    for name, returns in named_returns.items():
        labels = regime_labels.reindex(returns.index)
        for label in ["bull", "sideways", "crisis"]:
            subset = returns[labels == label].dropna()
            if len(subset) < 20:
                continue
            rows[(name, label)] = {
                "days": len(subset),
                "ann_return": float(subset.mean() * ANN),
                "ann_vol": float(subset.std() * np.sqrt(ANN)),
                "sharpe": sharpe(subset, min_observations=20),
                "hit_rate": float((subset > 0).mean()),
            }
    table = pd.DataFrame(rows).T
    table.index.names = ["strategy", "regime"]
    return table


def calendar_year_returns(named_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Total return per calendar year (rows = years, columns = names)."""

    frame = pd.DataFrame(named_returns)
    return frame.groupby(frame.index.year).apply(lambda block: (1.0 + block).prod() - 1.0)


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 252) -> pd.Series:
    """Rolling correlation between two return streams."""

    aligned = pd.concat([a, b], axis=1).dropna()
    return aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])


def pair_specification_grid(
    prices: pd.DataFrame,
    asset_returns: pd.DataFrame,
    calendar,
    *,
    transaction_cost_bps: float,
    slippage_bps: float,
    min_train_observations: int | None = None,
    test_observations: int | None = None,
) -> pd.DataFrame:
    """Compare the four ways of specifying a pair spread, walk-forward.

    Two choices are usually made silently and never revisited:

    * **levels or logs.** On price levels the OLS slope is "units of B per unit of A"
      and so depends on the price levels themselves. On log prices it is an elasticity,
      which is scale-free, closer to homoskedastic, and the standard formulation for
      cointegration on I(1) prices. Logs are the better-founded choice.
    * **unit or hedge-ratio-weighted legs.** The original rule estimates a hedge ratio
      to define the spread and then trades one unit of each leg, so the portfolio being
      traded is not the spread that was tested for stationarity. Weighting the legs by
      the hedge ratio makes the trade match the signal.

    Reporting all four is the point: if the sleeve's result depends on a choice that
    theory says should be innocuous, that is evidence about the sleeve, not about the
    choice.
    """

    import itertools

    import numpy as np

    from src.backtest.walk_forward import expanding_walk_forward_splits, training_sharpe
    from src.models.ou import compute_spread_zscore
    from src.utils.config import DATA_CONFIG, PAIRS_CONFIG, WALK_FORWARD_CONFIG

    if min_train_observations is None:
        min_train_observations = WALK_FORWARD_CONFIG.min_train_observations
    if test_observations is None:
        test_observations = WALK_FORWARD_CONFIG.test_observations
    splits = expanding_walk_forward_splits(calendar, min_train_observations, test_observations)
    pairs = PAIRS_CONFIG.candidate_pairs or list(itertools.combinations(sorted(DATA_CONFIG.symbols), 2))
    columns = sorted(DATA_CONFIG.symbols)

    def fold_positions(asset_a, asset_b, use_log, hedged):
        frame = pd.concat([prices[asset_a], prices[asset_b]], axis=1).dropna()
        frame.columns = ["a", "b"]
        chunks = []
        for train_index, test_index in splits:
            train = frame.reindex(train_index).dropna()
            if len(train) < 50:
                continue
            x = np.log(train["b"]) if use_log else train["b"]
            y = np.log(train["a"]) if use_log else train["a"]
            design = np.column_stack([np.ones(len(x)), x.values])
            alpha, beta = np.linalg.lstsq(design, y.values, rcond=None)[0]
            window = frame.reindex(train_index.union(test_index)).dropna()
            xf = np.log(window["b"]) if use_log else window["b"]
            yf = np.log(window["a"]) if use_log else window["a"]
            z = compute_spread_zscore(yf - alpha - beta * xf, window=PAIRS_CONFIG.zscore_window)
            signal = pd.Series(np.nan, index=z.index, dtype=float)
            signal[z > PAIRS_CONFIG.entry_threshold] = -1.0
            signal[z < -PAIRS_CONFIG.entry_threshold] = 1.0
            signal[z.abs() < PAIRS_CONFIG.exit_threshold] = 0.0
            signal[z.isna()] = np.nan
            signal = signal.ffill().fillna(0.0)
            if hedged:
                scale = 1.0 / (1.0 + abs(float(beta)))
                legs = pd.DataFrame({asset_a: signal * scale, asset_b: -signal * float(beta) * scale})
            else:
                legs = pd.DataFrame({asset_a: signal, asset_b: -signal})
            chunks.append(legs.reindex(test_index).fillna(0.0))
        return pd.concat(chunks).sort_index() if chunks else pd.DataFrame(columns=[asset_a, asset_b])

    rows = {}
    for label, (use_log, hedged) in {
        "levels, unit legs": (False, False),
        "logs, unit legs": (True, False),
        "levels, hedge-weighted": (False, True),
        "logs, hedge-weighted": (True, True),
    }.items():
        candidates, nets = {}, {}
        for asset_a, asset_b in pairs:
            positions = fold_positions(asset_a, asset_b, use_log, hedged)
            if positions.empty:
                continue
            candidates[f"{asset_a}-{asset_b}"] = positions
            nets[f"{asset_a}-{asset_b}"] = positions_to_strategy_returns(
                positions, asset_returns, transaction_cost_bps, slippage_bps)["net_return"]
        chunks, picks = [], {}
        for train_index, test_index in splits:
            scores = {k: training_sharpe(v.reindex(train_index)) for k, v in nets.items()}
            best = max(scores, key=scores.get)
            chunks.append(candidates[best].reindex(index=test_index, columns=columns).fillna(0.0))
            picks[test_index[0]] = best
        stitched = pd.concat(chunks).sort_index()
        stitched = stitched[~stitched.index.duplicated(keep="first")]
        backtest = positions_to_strategy_returns(
            stitched, asset_returns, transaction_cost_bps, slippage_bps)
        metrics = performance_metrics(backtest["net_return"])
        rows[label] = {
            "net_sharpe": metrics["sharpe_ratio"],
            "cagr": metrics["cagr"],
            "max_drawdown": metrics["max_drawdown"],
            "ann_turnover": float(backtest["turnover"].mean() * ANN),
            "most_selected_pair": pd.Series(picks).value_counts().index[0],
        }
    return pd.DataFrame(rows).T
