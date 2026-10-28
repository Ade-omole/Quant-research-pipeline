"""Monte Carlo risk analysis via a stationary block bootstrap.

Sampling individual days destroys volatility clustering, so we resample
*blocks* of consecutive days (with random block lengths, expected ~21 days).
Each simulated path is a plausible alternative year built from the strategy's
own return distribution, including its short-horizon autocorrelation.

The outputs answer questions a risk committee actually asks:
- what is the distribution of one-year outcomes?
- how bad can a drawdown plausibly get (not just how bad it was)?
- what are the Monte Carlo VaR and expected shortfall?
- what is the probability of losing money over the horizon?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import ROBUSTNESS_CONFIG


def stationary_block_bootstrap(
    returns: pd.Series,
    n_paths: int | None = None,
    horizon: int | None = None,
    avg_block: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate return paths. Returns an array of shape (n_paths, horizon).

    At every step a new block starts with probability 1/avg_block (jump to a
    random date); otherwise the path continues with the next historical day.
    The construction is fully vectorized.
    """

    n_paths = n_paths or ROBUSTNESS_CONFIG.mc_paths
    horizon = horizon or ROBUSTNESS_CONFIG.mc_horizon_days
    avg_block = avg_block or ROBUSTNESS_CONFIG.mc_avg_block
    seed = ROBUSTNESS_CONFIG.mc_seed if seed is None else seed

    values = returns.dropna().to_numpy(dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)

    # Where does a new block start? The first step always starts one.
    new_block = rng.random((n_paths, horizon)) < (1.0 / avg_block)
    new_block[:, 0] = True
    # Random historical start date for each potential block.
    starts = rng.integers(0, n, size=(n_paths, horizon))
    steps = np.arange(horizon)[None, :]
    # For each step, the time at which its current block began.
    block_start_time = np.maximum.accumulate(np.where(new_block, steps, -1), axis=1)
    # The historical index that block began at, plus the offset inside the block.
    block_start_index = np.take_along_axis(starts, block_start_time, axis=1)
    index = (block_start_index + (steps - block_start_time)) % n
    return values[index]


def _max_drawdown_of_paths(paths: np.ndarray) -> np.ndarray:
    equity = np.cumprod(1.0 + paths, axis=1)
    peaks = np.maximum.accumulate(equity, axis=1)
    return (equity / peaks - 1.0).min(axis=1)


def monte_carlo_summary(
    returns: pd.Series,
    n_paths: int | None = None,
    horizon: int | None = None,
    avg_block: int | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    """Distributional risk statistics over the simulation horizon."""

    paths = stationary_block_bootstrap(returns, n_paths, horizon, avg_block, seed)
    terminal = np.prod(1.0 + paths, axis=1) - 1.0
    max_drawdowns = _max_drawdown_of_paths(paths)
    var_threshold = np.quantile(terminal, 0.05)

    return {
        "horizon_days": float(paths.shape[1]),
        "n_paths": float(paths.shape[0]),
        "median_return": float(np.median(terminal)),
        "mean_return": float(np.mean(terminal)),
        "p5_return": float(var_threshold),
        "p95_return": float(np.quantile(terminal, 0.95)),
        "prob_loss": float((terminal < 0).mean()),
        "mc_var_95": float(-var_threshold),
        "mc_es_95": float(-terminal[terminal <= var_threshold].mean()),
        "median_max_drawdown": float(np.median(max_drawdowns)),
        "p95_max_drawdown": float(np.quantile(max_drawdowns, 0.05)),  # 5th percentile = worst tail
    }


def monte_carlo_table(named_returns: dict[str, pd.Series], **kwargs) -> pd.DataFrame:
    """Monte Carlo summary for several return streams side by side."""

    return pd.DataFrame({name: monte_carlo_summary(r, **kwargs) for name, r in named_returns.items()}).T


def monte_carlo_paths_frame(
    returns: pd.Series,
    n_paths: int = 200,
    horizon: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """A small set of simulated equity curves for plotting (fan chart)."""

    paths = stationary_block_bootstrap(returns, n_paths, horizon, seed=seed)
    equity = np.cumprod(1.0 + paths, axis=1)
    return pd.DataFrame(equity.T)  # rows = days, columns = paths
