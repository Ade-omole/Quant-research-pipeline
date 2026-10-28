"""Central configuration for symbols, dates, costs, and cache defaults.

Changes versus the original project
-----------------------------------
* The sample starts in 2004 instead of 2014. The asset list is unchanged, but the
  three youngest funds (FXE, USO, UUP) did not exist in 2004, so the price panel is
  *ragged*: each asset joins on its own inception date and the loader keeps rows in
  which at least the benchmark is present. This buys the 2008 crisis, 2011 and
  2015--16 without dropping or substituting any asset.
* A fourth strategy, adaptive asset allocation (AAA), is configured here.
* Walk-forward, portfolio, robustness and risk settings live here too, so every
  notebook and module reads one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- DataConfig: ---
# Remembers tickers, sample dates, benchmark vs pair, and rolling-window lengths.
# Notebooks and data_loader use this so every stage sees the same universe and calendar.
@dataclass(frozen=True)
class DataConfig:
    """Market dataset: symbols, dates, benchmark, pair, rolling windows."""

    # ETF universe used across the full pipeline (unchanged from the original project).
    # SPY = US equities, TLT = Treasuries, GLD = gold, USO = oil,
    # UUP = dollar index proxy, EFA/EEM = developed/emerging markets,
    # FXE = euro currency ETF.
    symbols: list[str] = field(
        default_factory=lambda: ["SPY", "TLT", "GLD", "USO", "UUP", "EFA", "EEM", "FXE"]
    )
    # Main asset for volatility timing and regime switching.
    benchmark_symbol: str = "SPY"
    # Historical sample range downloaded from Yahoo Finance.
    # 2004 is before three of the eight funds existed; see `allow_ragged_panel`.
    start_date: str = "2004-01-01"
    end_date: str = "2026-08-31"
    # Keep rows in which the benchmark trades even when younger assets are missing.
    # With this False the panel would silently start in March 2007 (UUP's inception).
    allow_ragged_panel: bool = True
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


@dataclass(frozen=True)
class AdaptiveAllocationConfig:
    """Adaptive asset allocation: momentum selection + inverse-volatility weighting.

    Each month, rank the ETF menu by 6-month total return, keep the top half,
    weight the survivors by inverse volatility, and move any selected asset whose
    own momentum is negative into the defensive asset instead.

    The menu is the project's own eight ETFs. TLT is the defensive asset because it
    is the only bond fund in this universe. `top_n = 4` is the canonical "top half"
    rule applied to eight assets; in the early ragged years fewer than eight assets
    exist and the rule ranks whatever is available.
    """

    # None means "use DATA_CONFIG.symbols", so the menu follows the project universe.
    menu: list[str] | None = None
    lookback: int = 126           # 6-month momentum window (trading days)
    top_n: int = 4                # hold the top half of an eight-asset menu
    vol_window: int = 63          # window for inverse-volatility sizing
    min_vol: float = 0.05         # floor on annualized vol so 1/vol stays bounded
    defensive_asset: str = "TLT"  # absolute-momentum safeguard parks capital here
    min_assets: int = 3           # do not trade until this many assets are rankable


@dataclass(frozen=True)
class PairsConfig:
    """Pairs trading: which pairs are eligible, and the entry/exit thresholds.

    The pair is no longer hard-coded. A full scan of the universe shows that only two
    of the 28 possible pairs reject "no cointegration" at the 5% level, which is what
    pure chance produces when 28 tests are run, and neither survives a Bonferroni
    correction. Worse, the cointegration test does not identify the tradable pairs:
    the pair with the best out-of-sample Sharpe fails the test with p = 0.90.

    Rather than pick a pair with hindsight, the walk-forward selects one each fold
    using training-window performance only (see ``walk_forward_pair_selection``).
    """

    # None means "every unordered pair of DATA_CONFIG.symbols".
    candidate_pairs: list[tuple[str, str]] | None = None
    entry_threshold: float = 1.5
    exit_threshold: float = 0.5
    zscore_window: int = 21
    # Regress log prices rather than price levels when forming the spread. Logs are the
    # better-founded choice (the hedge ratio becomes a scale-free elasticity and the
    # residual is closer to homoskedastic), and they do give slightly stronger
    # cointegration evidence here. They also produce a materially *worse* trading
    # result, which is reported as a robustness finding rather than resolved by
    # picking the specification that scores best. See `sensitivity_pair_spec`.
    use_log_prices: bool = False


@dataclass(frozen=True)
class WalkForwardConfig:
    """Expanding-window walk-forward settings shared by every strategy.

    Model-based strategies (random forest volatility, HMM regimes, pair hedge ratio)
    are refit on each expanding training window. AAA has no model to fit, so the
    walk-forward instead *selects its parameters* from a grid using training-window
    net Sharpe only.
    """

    min_train_observations: int = 504   # two years before the first test block
    test_observations: int = 21         # one month per test block
    aaa_lookbacks: tuple[int, ...] = (63, 126, 252)
    aaa_top_n: tuple[int, ...] = (3, 4, 5)
    # When True the pairs sleeve also picks its pair per fold from training data.
    select_pair_walk_forward: bool = True


@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio menus (which strategies) and weighting methods (how to combine them)."""

    # Pairs trading is deliberately absent. It is analysed in full as a rejected
    # strategy (see the spread-specification grid), but its apparent edge does not
    # survive changing the spread from price levels to log prices, or weighting the
    # legs by the hedge ratio that defines the spread. A result that moves that much
    # under innocuous implementation choices cannot be allocated capital.
    menus: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "Orig2": ("volatility_timing", "regime_switching"),
            "Core3": ("volatility_timing", "regime_switching", "adaptive_allocation"),
        }
    )
    methods: tuple[str, ...] = (
        "equal_weight",
        "inverse_volatility",
        "markowitz_baseline",
        "markowitz_concentration",
    )
    # Portfolio used for the headline figures and the Monte Carlo / drawdown studies.
    headline_menu: str = "Core3"
    headline_method: str = "inverse_volatility"


@dataclass(frozen=True)
class RobustnessConfig:
    """Sub-periods, cost grid, overlay grid, stress episodes, and Monte Carlo settings."""

    subperiods: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "2006-09": ("2006-01-01", "2009-12-31"),
            "2010-13": ("2010-01-01", "2013-12-31"),
            "2014-17": ("2014-01-01", "2017-12-31"),
            "2018-21": ("2018-01-01", "2021-12-31"),
            "2022-26": ("2022-01-01", "2026-12-31"),
        }
    )
    cost_grid_bps: tuple[float, ...] = (0.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)
    turnover_caps: tuple[float, ...] = (0.10, 0.20, 0.35, 0.50, 1.00)
    vol_targets: tuple[float, ...] = (0.08, 0.10, 0.12)
    stress_episodes: dict[str, tuple[str, str]] = field(
        default_factory=lambda: {
            "2008 global financial crisis": ("2007-10-01", "2009-03-09"),
            "2010 flash crash / euro scare": ("2010-04-15", "2010-07-06"),
            "2011 US downgrade / euro crisis": ("2011-07-01", "2011-10-03"),
            "2015 China deval / Aug flash crash": ("2015-08-01", "2015-09-30"),
            "2018 Feb VIX spike": ("2018-01-26", "2018-02-28"),
            "2018 Q4 rate-hike selloff": ("2018-10-01", "2018-12-31"),
            "2020 COVID crash": ("2020-02-15", "2020-04-15"),
            "2022 inflation / hiking cycle": ("2022-01-01", "2022-10-31"),
            "2025 tariff shock": ("2025-03-01", "2025-05-31"),
        }
    )
    mc_paths: int = 5000
    mc_horizon_days: int = 252
    mc_avg_block: int = 21
    mc_seed: int = 42


# These objects are imported by notebooks and modules.
# Creating them once keeps the rest of the project concise.
DATA_CONFIG = DataConfig()
COST_CONFIG = CostConfig()
AAA_CONFIG = AdaptiveAllocationConfig()
PAIRS_CONFIG = PairsConfig()
WALK_FORWARD_CONFIG = WalkForwardConfig()
PORTFOLIO_CONFIG = PortfolioConfig()
ROBUSTNESS_CONFIG = RobustnessConfig()

# Short display labels used in tables and figures.
STRATEGY_LABELS: dict[str, str] = {
    "volatility_timing": "VolTiming",
    "pairs_trading": "Pairs",
    "regime_switching": "RegimeHMM",
    "adaptive_allocation": "AAA",
}

# Strategies that are allocated capital. Pairs trading is built and analysed in full,
# because documenting why an idea is rejected is part of the research, but it is not
# carried into the portfolios or the downstream robustness and risk analysis.
RETAINED_STRATEGIES: tuple[str, ...] = (
    "volatility_timing", "regime_switching", "adaptive_allocation",
)
REJECTED_STRATEGIES: tuple[str, ...] = ("pairs_trading",)
