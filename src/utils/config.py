"""Central configuration for symbols, dates, costs, and cache defaults."""

from __future__ import annotations

from dataclasses import dataclass, field


# --- DataConfig: ---
# Remembers tickers, sample dates, benchmark vs pair, and rolling-window lengths.
# Notebooks and data_loader use this so every stage sees the same universe and calendar.
@dataclass(frozen=True)
class DataConfig:
    """Market dataset: symbols, dates, benchmark, pair, rolling windows."""

    # ETF universe used across the full pipeline.
    # SPY = US equities, TLT = Treasuries, GLD = gold, USO = oil,
    # UUP = dollar index proxy, EFA/EEM = developed/emerging markets,
    # FXE = euro currency ETF.
    symbols: list[str] = field(
        default_factory=lambda: ["SPY", "TLT", "GLD", "USO", "UUP", "EFA", "EEM", "FXE"]
    )
    # Default pair for stages 2–3 (change after pair scan in notebook 1).
    pair_symbols: tuple[str, str] = ("EFA", "EEM")
    # Main asset for volatility timing and regime switching.
    benchmark_symbol: str = "SPY"
    # Historical sample range downloaded from Yahoo Finance.
    start_date: str = "2014-01-01"
    end_date: str = "2025-12-31"
    # Daily trading data is commonly annualized with roughly 252 sessions.
    annualization_factor: int = 252
    # 21 trading days is approximately one month.
    rolling_vol_window: int = 21
    # 63 trading days is approximately one quarter.
    feature_window: int = 63
    # Local folder (relative to repo root) for cached adjusted-close prices.
    cache_dir: str = "data/cache"
    # When True, reuse cached prices instead of calling Yahoo on every run.
    use_price_cache: bool = True


@dataclass(frozen=True)
class CostConfig:
    """Trading friction and portfolio risk knobs."""

    # Trading fee assumption expressed in basis points.
    transaction_cost_bps: float = 5.0
    # Slippage assumption expressed in basis points.
    slippage_bps: float = 2.0
    # Intended rebalance cadence for future extensions.
    rebalance_frequency: str = "W-FRI"
    # Maximum allowed position change per rebalance.
    turnover_cap: float = 0.35
    # Target annualized volatility for the final portfolio.
    target_portfolio_vol: float = 0.10


# These objects are imported by notebooks and modules.
# Creating them once keeps the rest of the project concise.
DATA_CONFIG = DataConfig()
COST_CONFIG = CostConfig()
