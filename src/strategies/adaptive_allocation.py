"""Adaptive asset allocation (AAA): momentum selection + inverse-volatility weighting.

Idea: hold whatever has been working across the ETF menu, size the holdings so each
carries similar risk, and step out of anything that is falling.

Every month:

1. rank the menu by 6-month total return (relative momentum),
2. keep the top half,
3. weight the survivors by inverse volatility,
4. if a selected asset's own 6-month return is negative, move its weight to the
   defensive asset instead (absolute-momentum safeguard).

Nothing is estimated: every number is a deterministic function of trailing prices,
which makes the rule causal by construction. The one judgement call, the parameter
pair (lookback, top_n), is removed by the walk-forward selection in
``src/backtest/walk_forward.py``.

Ragged panel
------------
In the early years only four of the eight ETFs existed. Assets without enough price
history have a missing momentum and a missing volatility, so they are excluded from
the ranking and from the weight normalization automatically. The rule stays flat
until at least ``cfg.min_assets`` assets are rankable.
"""

from __future__ import annotations

import pandas as pd

from src.strategies.common import (
    annualized_rolling_volatility,
    first_date_with_min_assets,
    month_end_hold,
)
from src.utils.config import AAA_CONFIG, DATA_CONFIG, AdaptiveAllocationConfig


def resolve_menu(cfg: AdaptiveAllocationConfig = AAA_CONFIG) -> list[str]:
    """The asset menu: the configured list, or the project universe when unset."""

    return list(cfg.menu) if cfg.menu else list(DATA_CONFIG.symbols)


def build_positions(
    prices: pd.DataFrame,
    cfg: AdaptiveAllocationConfig = AAA_CONFIG,
) -> pd.DataFrame:
    """Return the daily AAA weight table (one column per menu asset), held monthly."""

    menu = [symbol for symbol in resolve_menu(cfg) if symbol in prices.columns]
    menu_prices = prices[menu]
    returns = menu_prices.pct_change(fill_method=None)

    # Step 1: relative momentum, the total return over the lookback window.
    # Assets that are too young to have a full window keep a missing value.
    momentum = menu_prices.pct_change(cfg.lookback, fill_method=None)

    # Step 2: rank each day (1 = strongest) and select the top N. Missing momentum
    # produces a missing rank, so a not-yet-listed asset can never be selected.
    rank = momentum.rank(axis=1, ascending=False)
    selected = (rank <= cfg.top_n).astype(float)

    # Step 3: inverse-volatility sizing among the selected assets. Multiplying by a
    # missing volatility yields a missing weight, which we then treat as no holding.
    inverse_volatility = 1.0 / annualized_rolling_volatility(returns, cfg.vol_window, cfg.min_vol)
    weights = (selected * inverse_volatility).fillna(0.0)
    weight_totals = weights.sum(axis=1)
    weights = weights.div(weight_totals.where(weight_totals > 0), axis=0).fillna(0.0)

    # Step 4: absolute-momentum safeguard. A selected asset whose own momentum is
    # negative hands its weight to the defensive asset rather than staying invested.
    #
    # The defensive asset can only absorb that weight once it is actually trading and
    # has enough history to be sized. Before then the weight is simply released to
    # cash, so the row sums to less than one. Parking capital in a fund that does not
    # exist yet would be a look-ahead error, and the backtest engine rejects it.
    falling = (momentum < 0) & (selected > 0)
    released = (weights * falling).sum(axis=1)
    weights = weights.where(~falling, 0.0)
    if cfg.defensive_asset in weights.columns:
        defensive_tradable = inverse_volatility[cfg.defensive_asset].notna()
        weights[cfg.defensive_asset] = weights[cfg.defensive_asset] + released.where(
            defensive_tradable, 0.0)

    # An asset can only be held on a date where it has a tradable return, which is a
    # second guard against holding a fund before its inception.
    weights = weights.where(returns.notna(), 0.0)

    # Stay flat until enough of the menu is rankable to make a selection meaningful.
    start = first_date_with_min_assets(momentum, cfg.min_assets)
    weights = weights.loc[start:]

    # Hold each month-end allocation until the next month-end.
    return month_end_hold(weights).fillna(0.0)


def describe_positions(positions: pd.DataFrame) -> pd.DataFrame:
    """Diagnostics for the write-up: how many assets are held and how often each is used."""

    held = positions > 1e-8
    return pd.DataFrame({
        "share_of_days_held": held.mean(),
        "average_weight_when_held": positions.where(held).mean(),
    }).sort_values("share_of_days_held", ascending=False)
