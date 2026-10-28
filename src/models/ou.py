"""Cointegration tests and Ornstein-Uhlenbeck spread estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def run_cointegration_test(series_a: pd.Series, series_b: pd.Series) -> dict[str, float]:
    """Engle-Granger cointegration test on aligned price levels.

    The panel is ragged: two ETFs may not have listed at the same time, so aligning
    on the index alone can leave missing values that the test cannot accept. Only the
    dates on which *both* series trade are used, and the number of those dates is
    returned so the caller can see how much history the test actually had.
    """

    aligned = pd.concat([series_a, series_b], axis=1, keys=["a", "b"]).dropna(how="any")
    if len(aligned) < 30:
        nan = float("nan")
        return {"test_statistic": nan, "p_value": nan, "critical_value_5pct": nan,
                "overlapping_observations": float(len(aligned))}

    test_statistic, p_value, critical_values = coint(aligned["a"], aligned["b"])
    return {
        "test_statistic": float(test_statistic),
        "p_value": float(p_value),
        "critical_value_5pct": float(critical_values[1]),
        "overlapping_observations": float(len(aligned)),
    }


def _estimate_ou_parameters_ols_delta(spread: pd.Series) -> dict[str, float]:
    """Estimate OU parameters by regressing Delta X on lagged X."""

    clean_spread = spread.dropna()
    lagged_spread = clean_spread.shift(1).dropna()
    spread_change = clean_spread.diff().dropna()
    aligned_lagged = lagged_spread.loc[spread_change.index]
    design_matrix = np.column_stack([np.ones(len(aligned_lagged)), aligned_lagged.values])
    alpha, beta = np.linalg.lstsq(design_matrix, spread_change.values, rcond=None)[0]

    mean_reversion_speed = max(1e-8, -beta)
    long_run_mean = -alpha / beta if beta != 0 else 0.0
    residuals = spread_change.values - design_matrix @ np.array([alpha, beta])
    residual_volatility = float(np.std(residuals))

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "kappa": float(mean_reversion_speed),
        "long_run_mean": float(long_run_mean),
        "residual_volatility": residual_volatility,
    }


def estimate_ou_parameters_exact_discrete(spread: pd.Series, *, dt: float = 1.0) -> dict[str, float]:
    """Exact discrete OU / AR(1) fit on spread levels."""

    x = spread.dropna().astype(float).to_numpy()
    if len(x) < 3:
        return _empty_exact_ou_result()

    x_prev = x[:-1]
    x_curr = x[1:]
    n = len(x_curr)
    design = np.column_stack([np.ones(n), x_prev])
    coef, _, _, _ = np.linalg.lstsq(design, x_curr, rcond=None)
    a_hat, b_hat = float(coef[0]), float(coef[1])
    fitted = design @ coef
    resid = x_curr - fitted
    sigma_eta_sq = float(np.mean(resid**2))
    sigma_eta = float(np.sqrt(sigma_eta_sq))

    if not (0.0 < b_hat < 1.0):
        return {
            "kappa": float("nan"),
            "long_run_mean": float("nan"),
            "sigma_ou": float("nan"),
            "phi": b_hat,
            "ar1_intercept": a_hat,
            "ar1_slope": b_hat,
            "residual_volatility": sigma_eta,
        }

    kappa = float(-np.log(b_hat) / dt)
    theta = float(a_hat / (1.0 - b_hat))
    sigma_ou_sq = 2.0 * kappa * sigma_eta_sq / (1.0 - b_hat**2)
    sigma_ou = float(np.sqrt(max(sigma_ou_sq, 0.0)))

    return {
        "kappa": kappa,
        "long_run_mean": theta,
        "sigma_ou": sigma_ou,
        "phi": b_hat,
        "ar1_intercept": a_hat,
        "ar1_slope": b_hat,
        "residual_volatility": sigma_eta,
    }


def _empty_exact_ou_result() -> dict[str, float]:
    nan = float("nan")
    return {
        "kappa": nan,
        "long_run_mean": nan,
        "sigma_ou": nan,
        "phi": nan,
        "ar1_intercept": nan,
        "ar1_slope": nan,
        "residual_volatility": nan,
    }


def estimate_ou_parameters(
    spread: pd.Series,
    *,
    method: str = "ols_delta",
    dt: float = 1.0,
) -> dict[str, float]:
    """``ols_delta`` or ``exact_discrete`` OU parameter estimation."""

    if method == "ols_delta":
        return _estimate_ou_parameters_ols_delta(spread)
    if method == "exact_discrete":
        return estimate_ou_parameters_exact_discrete(spread, dt=dt)
    raise ValueError(
        f"Unknown method {method!r}; use 'ols_delta' or 'exact_discrete'."
    )


def compute_spread_zscore(spread: pd.Series, window: int = 21) -> pd.Series:
    """Rolling z-score of the spread."""

    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore.rename("spread_zscore")
