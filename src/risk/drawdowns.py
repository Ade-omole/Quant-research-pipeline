"""Drawdown episode analysis: the largest peak-to-trough losses and how long they lasted."""

from __future__ import annotations

import pandas as pd

from src.backtest.engine import drawdown_curve, equity_curve


def drawdown_table(returns: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """The ``top_n`` deepest drawdowns with peak, trough, recovery dates and durations."""

    clean = returns.dropna()
    equity = equity_curve(clean)
    drawdown = drawdown_curve(equity)

    episodes: list[dict[str, object]] = []
    in_drawdown = False
    peak_date = None
    for date, value in drawdown.items():
        if not in_drawdown and value < 0:
            in_drawdown = True
            # The peak is the last date at which the drawdown was zero.
            peak_date = drawdown.loc[:date][drawdown.loc[:date] == 0].index[-1]
        elif in_drawdown and value == 0:
            in_drawdown = False
            segment = drawdown.loc[peak_date:date]
            trough_date = segment.idxmin()
            episodes.append({
                "peak": peak_date.date(),
                "trough": trough_date.date(),
                "recovery": date.date(),
                "depth": float(segment.min()),
                "days_to_trough": int(len(drawdown.loc[peak_date:trough_date]) - 1),
                "days_to_recover": int(len(drawdown.loc[trough_date:date]) - 1),
            })
    if in_drawdown:
        segment = drawdown.loc[peak_date:]
        trough_date = segment.idxmin()
        episodes.append({
            "peak": peak_date.date(),
            "trough": trough_date.date(),
            "recovery": "ongoing",
            "depth": float(segment.min()),
            "days_to_trough": int(len(drawdown.loc[peak_date:trough_date]) - 1),
            "days_to_recover": None,
        })

    table = pd.DataFrame(episodes)
    if table.empty:
        return table
    return table.sort_values("depth").head(top_n).reset_index(drop=True)
