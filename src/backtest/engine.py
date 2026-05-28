"""Reusable backtest functions for strategy and portfolio evaluation.

The backtest engine converts positions into returns and then converts returns
into performance statistics. Keeping this logic reusable means every strategy
is evaluated with the same accounting rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.execution.costs import compute_cost_rate, compute_turnover


def positions_to_strategy_returns(
    positions: pd.DataFrame,
    asset_returns: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """Convert positions into gross and net strategy returns.

    Positions are lagged by one day before multiplying by returns. This is a
    simple way to avoid look-ahead bias: today's signal can only earn tomorrow's
    return, not today's already-known return.
    """

    # Keep only dates that exist in both the position table and return table.
    aligned_positions, aligned_returns = positions.align(asset_returns, join="inner", axis=0)
    # Use yesterday's position to earn today's return.
    lagged_positions = aligned_positions.shift(1).fillna(0.0)  ## shift positions by one day; NaN -> 0
    gross_returns = (lagged_positions * aligned_returns).sum(axis=1).rename("gross_return")  # weight at t-1 times return at t
    # Turnover measures how much the position changed through time.
    turnover = compute_turnover(aligned_positions)
    # Costs are proportional to turnover, transaction-cost assumptions, and slippage.
    cost_rate = compute_cost_rate(turnover, transaction_cost_bps, slippage_bps)
    # Net return is what remains after subtracting trading costs.
    net_returns = (gross_returns - cost_rate).rename("net_return")

    # Return gross return, turnover, costs, and net return together.
    output = pd.concat([gross_returns, turnover, cost_rate, net_returns], axis=1)
    return output


def equity_curve(returns: pd.Series) -> pd.Series:
    """Compound returns into a growth-of-one-dollar series."""

    # Starting from 1.0, repeatedly multiply by (1 + return).
    return (1.0 + returns.fillna(0.0)).cumprod().rename("equity_curve")


def drawdown_curve(equity: pd.Series) -> pd.Series:
    """Track the percentage loss from the running peak."""

    # Running peak is the highest equity value observed so far.
    running_peak = equity.cummax()
    # Drawdown is negative when equity is below its previous high.
    drawdown = equity / running_peak - 1.0
    return drawdown.rename("drawdown")


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Estimate Value at Risk (VaR) using the historical-simulation approach.

    The historical VaR at confidence level c is the negative of the (1 - c)
    quantile of the return distribution. We report it as a positive number so
    that a daily VaR of 0.02 at 95 percent confidence reads as: on the worst
    5 percent of days the strategy is expected to lose at least 2 percent.
    Returning a positive loss makes risk reports easier to read at a glance.
    """

    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    tail_probability = 1.0 - confidence
    quantile_value = float(np.quantile(clean_returns, tail_probability))
    return float(-quantile_value)


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Estimate the Expected Shortfall (also called Conditional VaR).

    Whereas VaR only reports the threshold of the tail, Expected Shortfall
    averages the returns inside the tail. This makes it more conservative and
    more informative when the loss distribution has fat tails. As with VaR we
    report the result as a positive number representing an average loss.
    """

    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    tail_probability = 1.0 - confidence
    quantile_value = float(np.quantile(clean_returns, tail_probability))
    tail_returns = clean_returns[clean_returns <= quantile_value]
    if tail_returns.empty:
        return float(-quantile_value)
    return float(-tail_returns.mean())


def performance_metrics(returns: pd.Series, annualization_factor: int = 252) -> dict[str, float]:
    """Compute Sharpe, Sortino, CAGR, drawdown, Calmar, vol, VaR, and ES."""

    # Remove missing observations before computing statistics.
    clean_returns = returns.dropna()
    # If there are no returns, return zeros instead of raising an error.
    if clean_returns.empty:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "volatility": 0.0,
            "var_95": 0.0,
            "expected_shortfall_95": 0.0,
        }

    # Build the compounded equity path and drawdown path.
    equity = equity_curve(clean_returns)
    drawdown = drawdown_curve(equity)

    # Average daily return is the numerator of Sharpe and Sortino.
    average_return = clean_returns.mean()
    # Daily standard deviation is the risk estimate for Sharpe.
    volatility = clean_returns.std()
    # Downside volatility uses only negative returns for Sortino.
    downside_volatility = clean_returns[clean_returns < 0].std()

    # Annualized Sharpe scales daily mean/std by sqrt(252).
    sharpe_ratio = np.sqrt(annualization_factor) * average_return / volatility if volatility > 0 else 0.0
    # Sortino is similar to Sharpe but penalizes downside risk only.
    sortino_ratio = (
        np.sqrt(annualization_factor) * average_return / downside_volatility
        if downside_volatility and downside_volatility > 0
        else 0.0
    )

    # Convert the sample length into years for CAGR.
    total_years = max(len(clean_returns) / annualization_factor, 1 / annualization_factor)
    # CAGR is the annualized growth rate of the final equity value.
    cagr = equity.iloc[-1] ** (1 / total_years) - 1.0
    # Max drawdown is the worst percentage loss from a previous peak.
    max_drawdown = float(drawdown.min())
    # Calmar compares annual growth to worst drawdown.
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0
    # Tail-risk metrics describe the size of bad days, not their frequency.
    var_95 = historical_var(clean_returns, confidence=0.95)
    expected_shortfall_95 = expected_shortfall(clean_returns, confidence=0.95)

    return {
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "cagr": float(cagr),
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar_ratio),
        "volatility": float(volatility * np.sqrt(annualization_factor)),
        "var_95": float(var_95),
        "expected_shortfall_95": float(expected_shortfall_95),
    }


def rolling_sharpe_ratio(
    returns: pd.Series,
    window: int = 63,
    annualization_factor: int = 252,
) -> pd.Series:
    """Measure how stable performance is through time."""

    # Rolling mean estimates recent average return.
    rolling_mean = returns.rolling(window).mean()
    # Rolling standard deviation estimates recent volatility.
    rolling_std = returns.rolling(window).std()
    # Rolling Sharpe shows whether performance is stable or concentrated.
    rolling_sharpe = np.sqrt(annualization_factor) * rolling_mean / rolling_std
    return rolling_sharpe.rename("rolling_sharpe")
