"""Portfolio construction helpers for combining stage-two strategies.

At this layer, each strategy return stream is treated like an asset. The goal
is to combine imperfect strategies into a more stable portfolio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.backtest.engine import performance_metrics, positions_to_strategy_returns
from src.execution.costs import apply_turnover_cap
from src.utils.config import COST_CONFIG


CONCENTRATION_LAMBDA = 0.25

MARKOWITZ_VARIANTS: dict[str, dict[str, float | bool]] = {
    "markowitz_baseline": {"use_ledoit_wolf": False, "max_weight": 1.0, "concentration_lambda": 0.0},
    "markowitz_shrinkage": {"use_ledoit_wolf": True, "max_weight": 1.0, "concentration_lambda": 0.0},
    "markowitz_cap70": {"use_ledoit_wolf": False, "max_weight": 0.7, "concentration_lambda": 0.0},
    "markowitz_shrinkage_cap70": {"use_ledoit_wolf": True, "max_weight": 0.7, "concentration_lambda": 0.0},
    "markowitz_concentration": {
        "use_ledoit_wolf": False,
        "max_weight": 1.0,
        "concentration_lambda": CONCENTRATION_LAMBDA,
    },
}


def _estimate_covariance(
    window_returns: pd.DataFrame,
    annualization_factor: int,
    ridge_penalty: float,
    use_ledoit_wolf: bool,
) -> np.ndarray:
    """Annualized covariance from sample stats or Ledoit-Wolf shrinkage."""

    n_strategies = window_returns.shape[1]
    if use_ledoit_wolf:
        covariance = LedoitWolf().fit(window_returns.values).covariance_ * annualization_factor
    else:
        covariance = window_returns.cov().values * annualization_factor
    return covariance + np.eye(n_strategies) * ridge_penalty


def inverse_volatility_weights(strategy_returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Allocate more weight to strategies with lower trailing volatility.

    This is a simple risk-parity-style rule. Strategies that have recently been
    less volatile receive larger weights, and strategies that have recently
    been more volatile receive smaller weights.
    """

    # Estimate each strategy's trailing daily volatility.
    rolling_volatility = strategy_returns.rolling(window).std().clip(lower=1e-6)
    # Invert volatility so lower-volatility strategies get higher raw scores.
    raw_weights = 1.0 / rolling_volatility
    # Normalize each row so the weights across strategies sum to 1.
    normalized_weights = raw_weights.div(raw_weights.sum(axis=1), axis=0) ## what fraction of the total weight is each strategy? == raw_weights / sum(raw_weights)
    # Drop early rows before the rolling window has enough observations.
    return normalized_weights.dropna(how="any")


def markowitz_max_sharpe_weights(
    strategy_returns: pd.DataFrame,
    window: int = 252,
    annualization_factor: int = 252,
    ridge_penalty: float = 1e-6,
    *,
    use_ledoit_wolf: bool = False,
    max_weight: float = 1.0,
    concentration_lambda: float = 0.0,
) -> pd.DataFrame:
    """Rolling long-only max-Sharpe weights with optional shrinkage, caps, and concentration penalty."""

    if not 0.0 < max_weight <= 1.0:
        raise ValueError("max_weight must be in (0, 1].")
    n_strategies = len(strategy_returns.columns)
    if max_weight * n_strategies < 1.0 - 1e-12:
        raise ValueError("max_weight is too low for the number of strategies.")

    optimized_weights: list[pd.Series] = []
    strategy_names = list(strategy_returns.columns)
    equal_weights = np.repeat(1.0 / n_strategies, n_strategies)
    bounds = [(0.0, max_weight) for _ in strategy_names]
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}

    for end_position in range(window, len(strategy_returns) + 1):
        window_returns = strategy_returns.iloc[end_position - window : end_position]
        expected_returns = window_returns.mean().values * annualization_factor
        covariance = _estimate_covariance(
            window_returns,
            annualization_factor,
            ridge_penalty,
            use_ledoit_wolf,
        )

        def negative_objective(weights: np.ndarray) -> float:
            portfolio_return = float(weights @ expected_returns)
            portfolio_variance = float(weights @ covariance @ weights)
            portfolio_volatility = np.sqrt(max(portfolio_variance, ridge_penalty))
            sharpe = portfolio_return / portfolio_volatility
            concentration_penalty = concentration_lambda * float(np.sum(weights**2))
            return -sharpe + concentration_penalty

        result = minimize(
            negative_objective,
            equal_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            weights = np.clip(result.x, 0.0, max_weight)
            if np.isclose(max_weight, 1.0) and weights.sum() > 0:
                weights = weights / weights.sum()
        else:
            weights = equal_weights

        optimized_weights.append(
            pd.Series(weights, index=strategy_names, name=strategy_returns.index[end_position - 1])
        )

    return pd.DataFrame(optimized_weights)


def markowitz_variant_weights(
    strategy_returns: pd.DataFrame,
    variant: str,
    **kwargs: float | int | bool,
) -> pd.DataFrame:
    """Look up a named Markowitz specification from MARKOWITZ_VARIANTS."""

    if variant not in MARKOWITZ_VARIANTS:
        valid = ", ".join(sorted(MARKOWITZ_VARIANTS))
        raise ValueError(f"Unknown Markowitz variant {variant!r}. Choose from: {valid}")
    params = {**MARKOWITZ_VARIANTS[variant], **kwargs}
    return markowitz_max_sharpe_weights(strategy_returns, **params)


def apply_volatility_target(
    portfolio_returns: pd.Series,
    target_volatility: float,
    annualization_factor: int = 252,
    window: int = 63,
) -> pd.Series:
    """Scale the portfolio so realized volatility stays near the target.

    Volatility targeting increases exposure after calm periods and reduces
    exposure after volatile periods. The shift prevents using today's realized
    return to decide today's leverage.
    """

    # Estimate recent annualized portfolio volatility.
    realized_volatility = portfolio_returns.rolling(window).std() * np.sqrt(annualization_factor)
    # Leverage is target volatility divided by realized volatility.
    # It is clipped to avoid extreme leverage when realized volatility is tiny.
    leverage = (target_volatility / realized_volatility).clip(lower=0.0, upper=2.0)
    # Use yesterday's leverage to scale today's return and avoid look-ahead bias.
    scaled_returns = portfolio_returns * leverage.shift(1).fillna(1.0)
    return scaled_returns.rename("vol_targeted_return")


def combine_strategy_returns(
    strategy_returns: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    """Form a weighted portfolio from individual strategy return streams."""

    # Align weights and strategy returns by date before multiplying them.
    aligned_weights, aligned_returns = weights.align(strategy_returns, join="inner", axis=0)
    # Portfolio return is the weighted sum of individual strategy returns.
    combined_returns = (aligned_weights * aligned_returns).sum(axis=1)
    return combined_returns.rename("portfolio_return")


def run_markowitz_variants_pre_cost(strategy_returns: pd.DataFrame) -> dict[str, dict]:
    """Pre-cost Markowitz variants: optimize -> combine -> vol target.

    No turnover cap, transaction costs, or position lag.
    """

    results = {}
    for name in MARKOWITZ_VARIANTS:
        weights = markowitz_variant_weights(strategy_returns, name)
        portfolio_return = combine_strategy_returns(strategy_returns, weights)
        returns = apply_volatility_target(
            portfolio_return,
            COST_CONFIG.target_portfolio_vol,
        )
        results[name] = {"weights": weights, "returns": returns}
    return results


def run_markowitz_variants(strategy_returns: pd.DataFrame) -> dict[str, dict]:
    """Realistic Markowitz variants: optimize -> turnover cap -> costs -> vol target."""

    results = {}
    for name in MARKOWITZ_VARIANTS:
        weights = markowitz_variant_weights(strategy_returns, name)
        weights_capped = apply_turnover_cap(weights, COST_CONFIG.turnover_cap)
        backtest = positions_to_strategy_returns(
            weights_capped,
            strategy_returns,
            COST_CONFIG.transaction_cost_bps,
            COST_CONFIG.slippage_bps,
        )
        returns = apply_volatility_target(
            backtest["net_return"],
            COST_CONFIG.target_portfolio_vol,
        )
        results[name] = {
            "weights": weights,
            "weights_capped": weights_capped,
            "returns": returns,
        }
    return results


def markowitz_metrics_table(variant_results: dict[str, dict]) -> pd.DataFrame:
    """Sharpe/CAGR/etc. for each Markowitz variant on a shared calendar."""

    compare = pd.DataFrame(
        {name: res["returns"] for name, res in variant_results.items()}
    ).dropna(how="any")
    return pd.DataFrame(
        {col: performance_metrics(compare[col]) for col in compare.columns}
    )


def markowitz_avg_weights_table(variant_results: dict[str, dict]) -> pd.DataFrame:
    """Average % allocation per strategy; uses capped weights when available."""

    return pd.DataFrame(
        {
            name: (
                res["weights_capped"] if "weights_capped" in res else res["weights"]
            ).mean()
            * 100
            for name, res in variant_results.items()
        }
    ).round(1)
