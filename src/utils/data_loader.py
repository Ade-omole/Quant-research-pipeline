"""Download, cache, and engineer market data (prices, returns, spreads, features)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils.config import DATA_CONFIG


@dataclass
class MarketDataBundle:
    """Container for prices, log returns, and rolling realized volatility."""

    # Adjusted close price table, indexed by date and with one column per asset.
    prices: pd.DataFrame
    # Daily log returns computed from the price table (used by the models).
    log_returns: pd.DataFrame
    # Daily simple returns computed from the price table (used by the backtests).
    simple_returns: pd.DataFrame
    # Rolling annualized volatility computed from the return table.
    realized_volatility: pd.DataFrame


def project_root() -> Path:
    """Return the repository root (parent of ``src/``)."""

    return Path(__file__).resolve().parents[2]


def _price_cache_key(symbols: list[str], start_date: str, end_date: str) -> str:
    """Build a stable filename stem from the download request."""

    symbol_token = "-".join(sorted(symbols))
    return f"adjusted_close_{symbol_token}_{start_date}_{end_date}"


def _price_cache_paths(symbols: list[str], start_date: str, end_date: str) -> tuple[Path, Path]:
    """Return (prices_csv, metadata_json) paths for a download request."""

    cache_dir = project_root() / DATA_CONFIG.cache_dir
    stem = _price_cache_key(symbols, start_date, end_date)
    return cache_dir / f"{stem}.csv", cache_dir / f"{stem}.meta.json"


def _cache_request_metadata(symbols: list[str], start_date: str, end_date: str) -> dict[str, object]:
    """Metadata used to decide whether a cached file still matches the request."""

    return {
        "symbols": sorted(symbols),
        "start_date": start_date,
        "end_date": end_date,
    }


def _load_price_cache(
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """Load cached prices when the on-disk metadata matches the current request."""

    prices_path, meta_path = _price_cache_paths(symbols, start_date, end_date)
    if not prices_path.exists() or not meta_path.exists():
        return None

    with meta_path.open(encoding="utf-8") as meta_file:
        saved_metadata = json.load(meta_file)

    if saved_metadata.get("request") != _cache_request_metadata(symbols, start_date, end_date):
        return None

    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    prices.index.name = "Date"
    return prices.sort_index()


def _save_price_cache(
    prices: pd.DataFrame,
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> None:
    """Persist prices and request metadata to the local cache folder."""

    prices_path, meta_path = _price_cache_paths(symbols, start_date, end_date)
    prices_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(prices_path)

    metadata = {
        "request": _cache_request_metadata(symbols, start_date, end_date),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(prices)),
        "first_date": str(prices.index.min().date()),
        "last_date": str(prices.index.max().date()),
    }
    with meta_path.open("w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2)


def clear_price_cache() -> list[Path]:
    """Delete all cached price files. Returns the paths that were removed."""

    cache_dir = project_root() / DATA_CONFIG.cache_dir
    if not cache_dir.exists():
        return []

    removed_paths: list[Path] = []
    for path in cache_dir.glob("adjusted_close_*"):
        path.unlink()
        removed_paths.append(path)
    return removed_paths


def download_adjusted_close(
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    refresh: bool = False,
    use_cache: bool | None = None,
) -> pd.DataFrame:
    """Download adjusted closes from Yahoo Finance; cache under ``data/cache/`` by default."""

    # Use the provided inputs when available; otherwise fall back to defaults.
    use_symbols = symbols or DATA_CONFIG.symbols
    use_start = start_date or DATA_CONFIG.start_date
    use_end = end_date or DATA_CONFIG.end_date
    should_use_cache = DATA_CONFIG.use_price_cache if use_cache is None else use_cache

    if should_use_cache and not refresh:
        cached_prices = _load_price_cache(use_symbols, use_start, use_end)
        if cached_prices is not None:
            return cached_prices

    # `auto_adjust=True` means dividends and splits are already reflected.
    # That gives us a cleaner total-return-like price series for research.
    raw_data = yf.download(
        tickers=use_symbols,
        start=use_start,
        end=use_end,
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns a MultiIndex column layout when downloading many symbols.
    # In that case, we select the adjusted close columns for every ticker.
    if isinstance(raw_data.columns, pd.MultiIndex):
        price_frame = raw_data["Close"].copy()
    else:
        # For one symbol, yfinance returns a simple Series/DataFrame shape.
        # This branch makes the output look like the multi-symbol case.
        price_frame = raw_data.to_frame(name=use_symbols[0])

    # How much history we keep depends on whether a ragged panel is allowed.
    #
    # Three of the eight ETFs (FXE 2005, USO 2006, UUP 2007) did not exist in 2004.
    # `dropna(how="any")` would therefore silently truncate the whole sample to March
    # 2007, the inception of the youngest fund. With `allow_ragged_panel=True` we keep
    # every date on which the benchmark trades and let each asset appear on its own
    # inception date; columns are simply missing before then, and every downstream
    # calculation is written to tolerate that.
    price_frame = price_frame.sort_index()
    if DATA_CONFIG.allow_ragged_panel:
        # Keep a row when the benchmark trades; younger assets stay NaN until listed.
        benchmark = DATA_CONFIG.benchmark_symbol
        if benchmark in price_frame.columns:
            clean_prices = price_frame[price_frame[benchmark].notna()]
        else:
            clean_prices = price_frame.dropna(how="all")
    else:
        # Original behaviour: every asset must be present on every date.
        clean_prices = price_frame.dropna(how="any")  # how="any" drops any row with a NaN

    if should_use_cache:
        _save_price_cache(clean_prices, use_symbols, use_start, use_end)

    return clean_prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from prices."""

    # Log return is log(today's price / yesterday's price).
    # The first row becomes NaN because there is no previous price.
    log_returns = np.log(prices / prices.shift(1))
    # Drop rows that are entirely missing. Individual assets may still be missing
    # before their inception date; that is the ragged panel and is expected.
    return log_returns.dropna(how="all")


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple returns from prices.

    Simple returns are the right input for portfolio accounting, because a weighted
    sum of simple returns is exactly the simple return of the portfolio. Log returns
    are kept for the statistical models, where additivity through time is what matters.
    """

    simple_returns = prices / prices.shift(1) - 1.0
    return simple_returns.dropna(how="all")


def compute_realized_volatility(
    returns: pd.DataFrame,
    window: int | None = None,
    annualization_factor: int | None = None,
) -> pd.DataFrame:
    """Estimate rolling realized volatility from daily returns.

    Realized volatility is the historical standard deviation of returns over a
    recent window. We multiply by sqrt(252) to express it as annualized risk.
    """

    # Use project defaults if no custom window or annualization is supplied.
    use_window = window or DATA_CONFIG.rolling_vol_window
    use_annualization = annualization_factor or DATA_CONFIG.annualization_factor

    # Rolling standard deviation estimates daily volatility.
    # Annualization rescales daily volatility to a yearly number.
    realized_volatility = returns.rolling(use_window).std() * np.sqrt(use_annualization)
    return realized_volatility.dropna(how="all")


def build_market_data_bundle(*, refresh: bool = False, use_cache: bool | None = None) -> MarketDataBundle:
    """Download prices and build log returns, simple returns, realized volatility, and bundle."""

    # Step 1: download adjusted ETF prices.
    prices = download_adjusted_close(refresh=refresh, use_cache=use_cache)
    # Step 2: convert prices into daily log returns.
    log_returns = compute_log_returns(prices)
    # Step 3: convert prices into daily simple returns (used by every backtest).
    simple_returns = compute_simple_returns(prices)
    # Step 4: estimate rolling annualized volatility from log returns.
    realized_volatility = compute_realized_volatility(log_returns)
    # Step 5: return all four tables together.
    return MarketDataBundle(
        prices=prices,
        log_returns=log_returns,
        simple_returns=simple_returns,
        realized_volatility=realized_volatility,
    )


def asset_availability(prices: pd.DataFrame) -> pd.DataFrame:
    """First and last trading date of every asset, plus its share of the sample.

    With a ragged panel this table is the honest description of what the backtest
    actually had to work with, and it belongs in the write-up.
    """

    rows = {}
    total_rows = len(prices)
    for symbol in prices.columns:
        column = prices[symbol]
        rows[symbol] = {
            "first_date": str(column.first_valid_index().date()),
            "last_date": str(column.last_valid_index().date()),
            "observations": int(column.notna().sum()),
            "share_of_sample": float(column.notna().sum() / total_rows),
        }
    table = pd.DataFrame(rows).T
    return table.sort_values("first_date")


def make_volatility_features(
    returns: pd.Series,
    realized_volatility: pd.Series,
    window: int | None = None,
) -> pd.DataFrame:
    """Create a compact feature set for volatility prediction.

    Each row contains information known at time `t`. The target column is the
    next realized volatility value, so the model learns to forecast one step
    ahead.
    """

    # Use the default quarterly window unless the caller supplies another one.
    use_window = window or DATA_CONFIG.feature_window
    # Start an empty feature table with the same dates as the return series.
    feature_frame = pd.DataFrame(index=returns.index)
    # One-day lag: yesterday's market move.
    feature_frame["return_lag_1"] = returns.shift(1)
    # Five-day lag: roughly one trading week ago.
    feature_frame["return_lag_5"] = returns.shift(5)
    # Rolling average return gives a simple momentum/mean signal.
    feature_frame["rolling_mean"] = returns.rolling(use_window).mean()
    # Rolling standard deviation measures recent risk.
    feature_frame["rolling_std"] = returns.rolling(use_window).std()
    # Rolling skewness measures asymmetry of recent returns.
    feature_frame["rolling_skew"] = returns.rolling(use_window).skew()
    # Rolling kurtosis measures how fat-tailed recent returns were.
    feature_frame["rolling_kurt"] = returns.rolling(use_window).kurt()
    # Current realized volatility is often highly informative for future risk.
    feature_frame["realized_volatility"] = realized_volatility
    # Shift by -1 so today's row tries to predict the next volatility value.
    feature_frame["target_volatility"] = realized_volatility.shift(-1)
    # Drop rows with missing lagged/rolling values before fitting a model.
    return feature_frame.dropna(how="any")


def _pair_frame(series_a: pd.Series, series_b: pd.Series, use_log: bool) -> pd.DataFrame:
    """Align two price series and optionally take logs.

    Working in logs changes what the hedge ratio means. On price levels, the slope is
    "units of B per unit of A" and therefore depends on the price levels themselves, so
    it drifts as the two assets grow apart. On log prices the slope is an elasticity: a
    1% move in B corresponds to a beta% move in A. That is scale-free, which is both the
    standard formulation for cointegration on I(1) prices and the more stable quantity
    to estimate over a 20-year sample.
    """

    aligned = pd.concat([series_a, series_b], axis=1).dropna(how="any")
    aligned.columns = ["asset_a", "asset_b"]
    return np.log(aligned) if use_log else aligned


def estimate_pair_ols_coefficients(
    series_a: pd.Series, series_b: pd.Series, *, use_log: bool = False
) -> tuple[float, float]:
    """Return (intercept, hedge_ratio) from OLS of asset_a on a constant and asset_b."""

    aligned_prices = _pair_frame(series_a, series_b, use_log)
    design_matrix = np.column_stack([np.ones(len(aligned_prices)), aligned_prices["asset_b"]])
    intercept, hedge_ratio = np.linalg.lstsq(design_matrix, aligned_prices["asset_a"], rcond=None)[0]
    return float(intercept), float(hedge_ratio)


def spread_from_prices(
    series_a: pd.Series,
    series_b: pd.Series,
    intercept: float,
    hedge_ratio: float,
    *,
    use_log: bool = False,
) -> pd.Series:
    """Apply fixed OLS coefficients to prices and return the spread residual series.

    ``use_log`` must match the setting used to estimate the coefficients.
    """

    aligned_prices = _pair_frame(series_a, series_b, use_log)
    spread = aligned_prices["asset_a"] - intercept - hedge_ratio * aligned_prices["asset_b"]
    return spread.rename("spread")


def build_spread(
    series_a: pd.Series, series_b: pd.Series, *, use_log: bool | None = None
) -> pd.DataFrame:
    """Estimate a hedge ratio and construct the pair spread.

    The pair spread is the residual from a simple linear relationship:
    asset_a = intercept + hedge_ratio * asset_b + residual.
    If that residual mean-reverts, it may become a pairs-trading signal.

    Set ``use_log`` to regress log prices instead of levels; the default follows
    ``PAIRS_CONFIG.use_log_prices``.
    """

    from src.utils.config import PAIRS_CONFIG

    use_log = PAIRS_CONFIG.use_log_prices if use_log is None else use_log
    aligned_prices = _pair_frame(series_a, series_b, use_log)
    intercept, hedge_ratio = estimate_pair_ols_coefficients(series_a, series_b, use_log=use_log)
    spread = spread_from_prices(series_a, series_b, intercept, hedge_ratio, use_log=use_log)
    # Keep the raw prices and add the estimated spread and regression values.
    output = aligned_prices.copy()
    output["spread"] = spread  ## create a new column called spread        
    output["hedge_ratio"] = hedge_ratio  ## create a new column called hedge_ratio
    output["intercept"] = intercept  ## create a new column called intercept
    output["log_prices"] = use_log
    return output
