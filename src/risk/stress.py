"""Historical stress testing.

Rather than inventing shocks, we replay each strategy and portfolio through
the worst market episodes in the sample and report what it actually did.
This shows whether diversification held up exactly when it was needed.
"""

from __future__ import annotations

import pandas as pd

from src.backtest.engine import drawdown_curve, equity_curve
from src.utils.config import ROBUSTNESS_CONFIG


def episode_total_return(returns: pd.Series, start: str, end: str) -> float:
    """Compound return over an episode; NaN when the series has no data there."""

    window = returns.loc[start:end].dropna()
    if window.empty:
        return float("nan")
    return float((1.0 + window).prod() - 1.0)


def episode_max_drawdown(returns: pd.Series, start: str, end: str) -> float:
    """Worst peak-to-trough loss inside the episode."""

    window = returns.loc[start:end].dropna()
    if window.empty:
        return float("nan")
    return float(drawdown_curve(equity_curve(window)).min())


def stress_test(
    named_returns: dict[str, pd.Series],
    episodes: dict[str, tuple[str, str]] | None = None,
    statistic: str = "total_return",
) -> pd.DataFrame:
    """Episode-by-episode table (rows = episodes, columns = strategies)."""

    episodes = episodes or ROBUSTNESS_CONFIG.stress_episodes
    func = episode_total_return if statistic == "total_return" else episode_max_drawdown
    rows = {
        episode: {name: func(returns, start, end) for name, returns in named_returns.items()}
        for episode, (start, end) in episodes.items()
    }
    return pd.DataFrame(rows).T
