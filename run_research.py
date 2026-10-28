"""One-command reproduction of the full research pipeline.

Loads (or downloads) prices, builds the four strategies walk-forward, combines them
into portfolios, runs every robustness check (sub-periods, parameter grids, cost
sensitivity, overlay grid) and every risk analysis (stress episodes, block-bootstrap
Monte Carlo, drawdown episodes, regime conditioning), and writes all tables (CSV in
results/, LaTeX in results/tables/) and figures (PNG in results/).

Everything reported is walk-forward out of sample. There is no second "in-sample"
track: model-based strategies are refit on expanding training windows, and the
rule-based strategy has its parameters selected on training windows only.

    python run_research.py          # full run; the random-forest refits take a few minutes
    python run_research.py --fast   # reuse cached walk-forward positions from results/cache/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.analysis.reporting import metrics_table, save_table, style_axis
from src.analysis.robustness import (
    calendar_year_returns, cost_sensitivity, pair_specification_grid, parameter_grid,
    regime_conditioned_table, rolling_correlation, sharpe, subperiod_sharpes,
)
from src.backtest.engine import (
    drawdown_curve, equity_curve, performance_metrics, positions_to_strategy_returns,
    rolling_sharpe_ratio,
)
from src.backtest.walk_forward import (
    adaptive_allocation_candidates, infer_walk_forward_calendar, walk_forward_pair_selection,
    walk_forward_pairs_positions, walk_forward_regime_positions, walk_forward_rule_positions,
    walk_forward_volatility_positions,
)
from src.models.ou import compute_spread_zscore
from src.models.regime import build_regime_features, fit_hmm_regime_model, infer_regimes
from src.portfolio.construction import (
    METHOD_LABELS, build_portfolio, run_portfolio_grid, strategy_correlation,
)
from src.risk.drawdowns import drawdown_table
from src.risk.monte_carlo import monte_carlo_paths_frame, monte_carlo_table
from src.risk.stress import stress_test
from src.strategies import adaptive_allocation
from src.strategies.benchmarks import buy_and_hold_positions, sixty_forty_positions
from src.strategies.pairs_trading import pairs_positions_from_zscore
from src.strategies.regime_switching import map_regimes_to_labels, regime_based_positions
from src.strategies.volatility_timing import inverse_volatility_position
from src.utils.config import (
    AAA_CONFIG, COST_CONFIG, DATA_CONFIG, PAIRS_CONFIG, PORTFOLIO_CONFIG, REJECTED_STRATEGIES,
    RETAINED_STRATEGIES, ROBUSTNESS_CONFIG, STRATEGY_LABELS, WALK_FORWARD_CONFIG,
)
from src.utils.data_loader import (
    asset_availability, build_market_data_bundle, build_spread, make_volatility_features,
)

RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
# LaTeX-ready copies of every table, as a by-product of the run. The research
# summaries build their own tables from results/ via make_summary_assets.py.
TABLES = RESULTS / "tables"
ANN = DATA_CONFIG.annualization_factor
TC, SLIP = COST_CONFIG.transaction_cost_bps, COST_CONFIG.slippage_bps
STRATEGY_ORDER = list(STRATEGY_LABELS)          # every strategy that is built
RETAINED_ORDER = list(RETAINED_STRATEGIES)      # the ones that receive capital
COLORS = {
    "VolTiming": "tab:blue", "Pairs": "tab:gray", "RegimeHMM": "tab:purple",
    "AAA": "tab:green", "SPY": "black", "60/40": "dimgray",
}


def save(frame: pd.DataFrame, name: str, float_fmt: str = "%.3f", **kwargs) -> None:
    save_table(frame, name, RESULTS, TABLES, float_fmt, **kwargs)


def net(positions, asset_returns, tc=TC, slip=SLIP):
    return positions_to_strategy_returns(positions, asset_returns, tc, slip)


def label(name: str) -> str:
    return STRATEGY_LABELS.get(name, name)


def relabel(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(index=STRATEGY_LABELS, columns=STRATEGY_LABELS)


def plot_equity_and_drawdown(named, title, path, log_scale=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    for name, series in named.items():
        eq = equity_curve(series.dropna())
        color = COLORS.get(name)
        lw = 1.8 if name in ("SPY", "60/40") else 1.3
        ax1.plot(eq, label=name, lw=lw, color=color, alpha=0.9)
        ax2.plot(drawdown_curve(eq), lw=1.0, color=color, alpha=0.9)
    if log_scale:
        ax1.set_yscale("log")
    ax1.set_title(title)
    ax1.set_ylabel("Growth of $1" + (" (log scale)" if log_scale else ""))
    ax1.legend(ncol=3, fontsize=9)
    ax2.set_title("Drawdown from peak")
    ax2.set_ylabel("Drawdown")
    for ax in (ax1, ax2):
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmap(frame, title, path, fmt="%.2f", cmap="RdBu_r", vmin=None, vmax=None, figsize=(7, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    values = frame.values.astype(float)
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(frame.shape[1]))
    ax.set_xticklabels(frame.columns, rotation=45, ha="right")
    ax.set_yticks(range(frame.shape[0]))
    ax.set_yticklabels(frame.index)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            v = values[i, j]
            ax.text(j, i, "--" if np.isnan(v) else fmt % v, ha="center", va="center", fontsize=9)
    fig.colorbar(im, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="reuse cached walk-forward positions")
    args = parser.parse_args()
    for folder in (RESULTS, CACHE, TABLES):
        folder.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("=== 1. Data ===")
    bundle = build_market_data_bundle()
    prices, log_returns, R = bundle.prices, bundle.log_returns, bundle.simple_returns
    realized_vol = bundle.realized_volatility
    bench = DATA_CONFIG.benchmark_symbol
    bench_log = log_returns[bench]
    bench_vol = realized_vol[bench].reindex(bench_log.index)
    print(f"  {prices.shape[1]} symbols, {prices.index.min().date()} to {prices.index.max().date()}")

    availability = asset_availability(prices)
    save(availability, "asset_availability", "%.3f", column_formats={"observations": "%.0f"})
    print(availability.to_string())

    spy = net(buy_and_hold_positions(R.index, bench), R)["net_return"].rename("SPY")
    sixty_forty = net(sixty_forty_positions(R.index, bench, "TLT"), R)["net_return"].rename("60/40")

    # ------------------------------------------------------------------
    # 2. Signals shared by the strategies
    # ------------------------------------------------------------------
    print("=== 2. Signals ===")
    volatility_features = make_volatility_features(bench_log, bench_vol)
    regime_features = build_regime_features(bench_log, bench_vol)
    # The full-sample HMM labels are used only to *describe* regimes in the risk
    # section; the tradable regime strategy is refit walk-forward below.
    hmm_model = fit_hmm_regime_model(regime_features)
    regime_labels = map_regimes_to_labels(
        infer_regimes(hmm_model, regime_features), bench_log)["regime_label"]

    aaa_reference = adaptive_allocation.build_positions(prices)
    candidates = adaptive_allocation_candidates(prices)
    calendar = infer_walk_forward_calendar(
        volatility_features, regime_features, extra_indexes=[aaa_reference.index])
    print(f"  shared walk-forward calendar: {calendar.min().date()} to {calendar.max().date()} "
          f"({len(calendar)} days)")

    # ------------------------------------------------------------------
    # 3. Walk-forward strategies (the only track)
    # ------------------------------------------------------------------
    print("=== 3. Walk-forward strategies ===")
    wf_kwargs = dict(min_train_observations=WALK_FORWARD_CONFIG.min_train_observations,
                     test_observations=WALK_FORWARD_CONFIG.test_observations)
    cache_files = {name: CACHE / f"wf_positions_{name}.csv" for name in STRATEGY_ORDER}
    choices_file = CACHE / "wf_parameter_choices.csv"

    if args.fast and all(f.exists() for f in cache_files.values()) and choices_file.exists():
        positions = {n: pd.read_csv(f, index_col=0, parse_dates=True) for n, f in cache_files.items()}
        wf_choices = pd.read_csv(choices_file, index_col=0, parse_dates=True)
        print("  reused cached walk-forward positions")
    else:
        positions = {}
        t = time.time()
        positions["volatility_timing"] = walk_forward_volatility_positions(
            volatility_features, bench, calendar=calendar, **wf_kwargs)
        print(f"  random forest refits done in {time.time() - t:.0f}s")
        t = time.time()
        positions["regime_switching"] = walk_forward_regime_positions(
            regime_features, bench_log, bench, calendar=calendar, **wf_kwargs)
        print(f"  HMM refits done in {time.time() - t:.0f}s")
        if WALK_FORWARD_CONFIG.select_pair_walk_forward:
            # The pair itself is chosen per fold on training data. Hard-coding one pair
            # would be a hindsight decision, and the cointegration scan gives no honest
            # basis for choosing: only 2 of 28 pairs pass at 5%, which is chance-level,
            # and the best out-of-sample pair fails the test outright.
            t = time.time()
            positions["pairs_trading"], pair_choices = walk_forward_pair_selection(
                prices, R, calendar=calendar, transaction_cost_bps=TC, slippage_bps=SLIP, **wf_kwargs)
            print(f"  pair selection over all eligible pairs done in {time.time() - t:.0f}s")
        else:
            first, second = sorted(DATA_CONFIG.symbols)[:2]
            positions["pairs_trading"] = walk_forward_pairs_positions(
                prices[first], prices[second], first, second, calendar=calendar,
                zscore_window=PAIRS_CONFIG.zscore_window, **wf_kwargs)
            pair_choices = pd.Series(f"{first}-{second}", index=positions["pairs_trading"].index[:1])
        positions["adaptive_allocation"], choice_series = walk_forward_rule_positions(
            candidates, R, calendar=calendar, transaction_cost_bps=TC, slippage_bps=SLIP, **wf_kwargs)
        wf_choices = pd.DataFrame({"adaptive_allocation": choice_series,
                                   "pairs_trading": pair_choices})
        for name, path in cache_files.items():
            positions[name].to_csv(path)
        wf_choices.to_csv(choices_file)

    backtests = {name: net(pos, R) for name, pos in positions.items()}
    strategies = pd.DataFrame({name: bt["net_return"] for name, bt in backtests.items()})
    oos_start = strategies.dropna(how="all").index.min()
    for name in STRATEGY_ORDER:
        print(f"  {label(name):10s} OOS Sharpe {sharpe(strategies[name]):5.2f}   "
              f"turnover {backtests[name]['turnover'].mean() * ANN:5.1f}x/yr")
    print(f"  OOS window: {oos_start.date()} to {strategies.index.max().date()}")

    named = {label(n): strategies[n] for n in STRATEGY_ORDER}
    # Downstream analysis uses only the retained strategies; pairs trading is reported
    # at strategy level as a rejected candidate and goes no further.
    named_retained = {label(n): strategies[n] for n in RETAINED_ORDER}
    benchmarks = {"SPY": spy.loc[oos_start:], "60/40": sixty_forty.loc[oos_start:]}

    # ------------------------------------------------------------------
    # 4. Strategy metrics
    # ------------------------------------------------------------------
    print("=== 4. Strategy metrics ===")
    m_strategies = metrics_table({**named, **benchmarks})
    m_strategies.insert(len(m_strategies.columns) - 2, "ann_turnover", pd.Series(
        {label(n): float(backtests[n]["turnover"].mean() * ANN) for n in STRATEGY_ORDER}))
    save(m_strategies, "metrics_strategies",
         column_formats={"years": "%.1f", "ann_turnover": "%.1f"})
    print(m_strategies[["sharpe_ratio", "cagr", "max_drawdown", "calmar_ratio"]].round(3).to_string())

    choice_counts = wf_choices["adaptive_allocation"].dropna().value_counts(normalize=True)
    save(choice_counts.rename("share_of_folds").to_frame(), "wf_parameter_choices", "%.3f")
    if "pairs_trading" in wf_choices:
        pair_counts = wf_choices["pairs_trading"].dropna().value_counts(normalize=True)
        save(pair_counts.rename("share_of_folds").to_frame().head(10), "wf_pair_choices", "%.3f")
        print("  pairs sleeve, most-selected pairs: "
              + ", ".join(f"{k} {v:.0%}" for k, v in pair_counts.head(3).items()))

    aaa_usage = adaptive_allocation.describe_positions(positions["adaptive_allocation"])
    save(aaa_usage, "aaa_asset_usage", "%.3f")

    # ------------------------------------------------------------------
    # 4b. Why pairs trading is rejected
    # ------------------------------------------------------------------
    # The sleeve's apparent edge should survive choices that theory says are close to
    # innocuous: regressing log prices rather than levels, and weighting the legs by
    # the hedge ratio that defines the spread. It does not, which is the evidence for
    # excluding it from every portfolio below.
    print("=== 4b. Pairs trading: spread specification grid ===")
    pair_spec = pair_specification_grid(
        prices, R, calendar, transaction_cost_bps=TC, slippage_bps=SLIP, **wf_kwargs)
    save(pair_spec, "sensitivity_pair_spec", "%.3f", column_formats={"ann_turnover": "%.1f"})
    print(pair_spec[["net_sharpe", "max_drawdown", "ann_turnover", "most_selected_pair"]]
          .round(3).to_string())
    print(f"  spread of outcomes across specifications: "
          f"{pair_spec['net_sharpe'].astype(float).max() - pair_spec['net_sharpe'].astype(float).min():.2f} "
          f"Sharpe; the sleeve is excluded from all portfolios.")

    # ------------------------------------------------------------------
    # 5. Correlations
    # ------------------------------------------------------------------
    print("=== 5. Correlations ===")
    corr = relabel(strategy_correlation(strategies[RETAINED_ORDER]))
    corr_all = relabel(strategy_correlation(strategies))
    save(corr_all, "strategy_correlations_all", "%.2f")
    save(corr, "strategy_correlations", "%.2f")
    heatmap(corr, "Strategy return correlations (walk-forward, common sample)",
            RESULTS / "fig_correlations.png", vmin=-1, vmax=1)
    corr_spy = pd.DataFrame({"corr_to_SPY": strategies[RETAINED_ORDER].corrwith(spy),
                             "corr_to_60/40": strategies[RETAINED_ORDER].corrwith(sixty_forty)})
    save(relabel(corr_spy), "correlations_to_benchmarks", "%.2f")
    print(corr.round(2).to_string())

    # ------------------------------------------------------------------
    # 6. Portfolios
    # ------------------------------------------------------------------
    print("=== 6. Portfolios ===")
    menus, methods = PORTFOLIO_CONFIG.menus, PORTFOLIO_CONFIG.methods
    t = time.time()
    grid, portfolios, results = run_portfolio_grid(strategies, menus, methods)
    print(f"  {len(portfolios)} portfolios in {time.time() - t:.0f}s")

    tidy = grid[["menu", "method", "sharpe_ratio", "sortino_ratio", "cagr", "volatility",
                 "max_drawdown", "calmar_ratio", "expected_shortfall_95", "ann_turnover", "start"]].copy()
    tidy["method"] = tidy["method"].map(METHOD_LABELS)
    for column in tidy.columns:
        if column not in ("menu", "method", "start"):
            tidy[column] = tidy[column].astype(float)
    save(tidy, "metrics_portfolios_all", column_formats={"ann_turnover": "%.1f"})

    def pivot(metric: str) -> pd.DataFrame:
        table = grid.pivot(index="menu", columns="method", values=metric).astype(float)
        return table.reindex(index=list(menus), columns=list(methods)).rename(columns=METHOD_LABELS)

    sharpe_grid, maxdd_grid = pivot("sharpe_ratio"), pivot("max_drawdown")
    save(sharpe_grid, "portfolio_grid_sharpe", "%.2f")
    save(maxdd_grid, "portfolio_grid_maxdd", "%.3f")

    headline_name = f"{PORTFOLIO_CONFIG.headline_menu}_{PORTFOLIO_CONFIG.headline_method}"
    headline = portfolios[headline_name]
    port_start = headline.index.min()
    key_ports = {
        "Orig2 inv-vol": portfolios["Orig2_inverse_volatility"],
        "Orig2 equal": portfolios["Orig2_equal_weight"],
        "Core3 inv-vol": headline,
        "Core3 equal": portfolios["Core3_equal_weight"],
        "Core3 Markowitz reg.": portfolios["Core3_markowitz_concentration"],
    }
    key_benchmarks = {"SPY": spy.loc[port_start:], "60/40": sixty_forty.loc[port_start:]}
    m_ports = metrics_table({**key_ports, **key_benchmarks})
    save(m_ports, "metrics_portfolios_key", column_formats={"years": "%.1f"})
    print(m_ports[["sharpe_ratio", "cagr", "max_drawdown", "calmar_ratio"]].round(3).to_string())

    avg_weights = pd.DataFrame({
        f"{menu} {METHOD_LABELS[method]}": results[f"{menu}_{method}"]["weights_capped"].mean() * 100
        for menu in menus for method in ("inverse_volatility", "markowitz_concentration")
    }).T
    save(relabel(avg_weights), "portfolio_avg_weights", "%.1f")

    # ------------------------------------------------------------------
    # 7. Robustness
    # ------------------------------------------------------------------
    print("=== 7. Robustness ===")
    sub_strategies = subperiod_sharpes({**named_retained, **benchmarks})
    save(sub_strategies, "subperiod_sharpes_strategies", "%.2f")
    sub_ports = subperiod_sharpes({**key_ports, **key_benchmarks})
    save(sub_ports, "subperiod_sharpes_portfolios", "%.2f")

    # Cost sensitivity keeps pairs trading, because its break-even cost is part of the
    # evidence for rejecting it.
    cost_strategies = cost_sensitivity({label(n): positions[n] for n in STRATEGY_ORDER}, R)
    cost_strategies["assumed_bps"] = TC + SLIP
    save(cost_strategies, "cost_sensitivity_strategies", "%.2f")

    cost_rows = {}
    for bps in ROBUSTNESS_CONFIG.cost_grid_bps:
        rets_bps = pd.DataFrame({n: net(positions[n], R, bps, 0.0)["net_return"] for n in STRATEGY_ORDER})
        cost_rows[f"{bps:g}bps"] = {
            f"{menu} inv-vol": sharpe(build_portfolio(
                rets_bps[list(menus[menu])].dropna(), "inverse_volatility",
                transaction_cost_bps=bps, slippage_bps=0.0)["returns"])
            for menu in menus}
    cost_ports = pd.DataFrame(cost_rows).T
    save(cost_ports, "cost_sensitivity_portfolios", "%.2f")

    grid_aaa = parameter_grid(prices, R, adaptive_allocation.build_positions, AAA_CONFIG,
                              {"top_n": [2, 3, 4, 5, 6], "lookback": [63, 126, 189, 252]},
                              TC, SLIP, start=oos_start)
    grid_aaa.columns = [c.replace("lookback=", "lb ") for c in grid_aaa.columns]
    grid_aaa.index = [i.replace("top_n=", "top ") for i in grid_aaa.index]
    save(grid_aaa, "sensitivity_aaa", "%.2f")

    grid_aaa_vol = parameter_grid(prices, R, adaptive_allocation.build_positions, AAA_CONFIG,
                                  {"vol_window": [21, 42, 63, 126], "defensive_asset": ["TLT", "GLD"]},
                                  TC, SLIP, start=oos_start)
    grid_aaa_vol.columns = [c.replace("defensive_asset=", "defensive ") for c in grid_aaa_vol.columns]
    grid_aaa_vol.index = [i.replace("vol_window=", "vol ") for i in grid_aaa_vol.index]
    save(grid_aaa_vol, "sensitivity_aaa_vol", "%.2f")

    # The originals each have one free parameter; vary it on the same OOS window.
    original_rows = {}
    vol_signal_proxy = positions["volatility_timing"][bench]
    for leverage in (1.0, 1.25, 1.5, 2.0):
        capped = vol_signal_proxy.clip(upper=leverage).to_frame(name=bench)
        series = net(capped, R)["net_return"].loc[oos_start:]
        original_rows[f"VolTiming leverage cap={leverage}"] = {
            "sharpe": sharpe(series), "max_drawdown": performance_metrics(series)["max_drawdown"]}
    for sideways in (0.0, 0.2, 0.35, 0.5, 0.7):
        weights = regime_labels.map({"bull": 1.0, "sideways": sideways, "crisis": 0.0}).fillna(0.0)
        series = net(weights.to_frame(name=bench), R)["net_return"].loc[oos_start:]
        original_rows[f"RegimeHMM sideways weight={sideways}"] = {
            "sharpe": sharpe(series), "max_drawdown": performance_metrics(series)["max_drawdown"]}
    # Entry/exit thresholds are varied on the pair the walk-forward actually traded most.
    top_pair = wf_choices["pairs_trading"].dropna().value_counts().index[0]
    leg_a, leg_b = top_pair.split("-")
    zscore = compute_spread_zscore(
        build_spread(prices[leg_a], prices[leg_b])["spread"], window=PAIRS_CONFIG.zscore_window)
    for entry, exit_ in ((1.0, 0.25), (1.5, 0.5), (2.0, 0.5), (2.5, 0.5)):
        series = net(pairs_positions_from_zscore(zscore, leg_a, leg_b, entry, exit_),
                     R)["net_return"].loc[oos_start:]
        original_rows[f"Pairs ({top_pair}) entry={entry}, exit={exit_}"] = {
            "sharpe": sharpe(series), "max_drawdown": performance_metrics(series)["max_drawdown"]}
    save(pd.DataFrame(original_rows).T, "sensitivity_originals", "%.2f")

    core3 = strategies[list(menus["Core3"])].dropna()
    overlay_rows = {}
    for cap in ROBUSTNESS_CONFIG.turnover_caps:
        row = {}
        for target in ROBUSTNESS_CONFIG.vol_targets:
            metrics = performance_metrics(build_portfolio(
                core3, "inverse_volatility", turnover_cap=cap, target_volatility=target)["returns"])
            row[f"vol {target:.0%} Sharpe"] = metrics["sharpe_ratio"]
            row[f"vol {target:.0%} MaxDD"] = metrics["max_drawdown"]
        overlay_rows[f"cap {cap:.2f}"] = row
    overlay = pd.DataFrame(overlay_rows).T
    save(overlay, "overlay_sensitivity", "%.2f")

    # ------------------------------------------------------------------
    # 8. Risk analysis
    # ------------------------------------------------------------------
    print("=== 8. Risk ===")
    stress_named = {**named_retained, "Core3 inv-vol": headline, "SPY": spy, "60/40": sixty_forty}
    stress = stress_test(stress_named)
    save(stress, "stress_tests", "%.3f")
    save(stress_test(stress_named, statistic="max_drawdown"), "stress_tests_drawdown", "%.3f")
    print(stress.round(3).to_string())

    mc_named = {**key_ports, "AAA": strategies["adaptive_allocation"], **key_benchmarks}
    mc = monte_carlo_table(mc_named)
    save(mc, "monte_carlo_summary", "%.3f",
         column_formats={"horizon_days": "%.0f", "n_paths": "%.0f"})

    for series, name in ((headline, "drawdown_table_headline"), (spy.loc[port_start:], "drawdown_table_spy")):
        table = drawdown_table(series.dropna(), top_n=5)
        table.index = [f"#{i + 1}" for i in range(len(table))]
        save(table, name, "%.3f")

    regime_conditioned = regime_conditioned_table(
        {**named_retained, "Core3 inv-vol": headline, "SPY": spy.loc[oos_start:]}, regime_labels)
    save(regime_conditioned, "regime_conditioned", "%.3f", column_formats={"days": "%.0f"})
    save(regime_labels.loc[oos_start:].value_counts(normalize=True)
         .rename("share_of_days").to_frame(), "regime_shares", "%.2f")

    calendar_returns = calendar_year_returns({**key_ports, **key_benchmarks})
    save(calendar_returns, "calendar_year_returns", "%.3f")

    # ------------------------------------------------------------------
    # 9. Figures
    # ------------------------------------------------------------------
    print("=== 9. Figures ===")
    plot_equity_and_drawdown({**named, **benchmarks},
                             "Walk-forward out-of-sample strategies, net of costs "
                             "(Pairs shown but not allocated capital)",
                             RESULTS / "fig_strategy_equity.png")
    plot_equity_and_drawdown({**key_ports, **key_benchmarks},
                             "Walk-forward portfolios versus benchmarks, after costs, 10% vol target",
                             RESULTS / "fig_portfolio_equity.png")

    weights = positions["adaptive_allocation"]
    weights = weights.loc[:, weights.abs().sum() > 0]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.stackplot(weights.index, [weights[c].values for c in weights.columns],
                 labels=list(weights.columns), alpha=0.85)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Portfolio weight")
    ax.set_title("AAA: asset weights over time (walk-forward); the gap below 1.0 is cash")
    ax.legend(ncol=8, fontsize=8, loc="upper left")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_aaa_weights.png", dpi=150)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.4]})
    ax1.plot(prices.notna().sum(axis=1), lw=1.4, color="tab:blue")
    ax1.set_ylim(0, 9)
    ax1.set_ylabel("assets trading")
    ax1.set_title("Ragged panel: number of ETFs available (GLD 2004, FXE 2005, USO 2006, UUP 2007)")
    held = (positions["adaptive_allocation"] > 1e-8).sum(axis=1)
    ax2.plot(held, lw=1.0, color="tab:green")
    ax2.set_ylabel("assets held by AAA")
    ax2.set_title("AAA: number of assets held (selection plus the defensive parking asset)")
    for ax in (ax1, ax2):
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_universe_availability.png", dpi=150)
    plt.close(fig)

    cats = list(candidates)
    codes = wf_choices["adaptive_allocation"].dropna().map({c: i for i, c in enumerate(cats)})
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.step(codes.index, codes.values, where="post", lw=1.3)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=8)
    ax.set_title("AAA: parameter set selected at each walk-forward fold (by training-window Sharpe)")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_wf_param_choices.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    # The drawdown panel is drawn in percent, to one decimal, so that it matches the
    # drawdown table cell for cell. In fraction units at two decimals, values that
    # genuinely differ (-23.8% and -23.5%) both render as "-0.24".
    for ax, (table, title, vmin, vmax, fmt) in zip(axes, (
            (sharpe_grid, "Net Sharpe", 0.0, 1.2, "%.2f"),
            (maxdd_grid * 100.0, "Maximum drawdown (%)", -35.0, 0.0, "%.1f"))):
        values = table.values.astype(float)
        im = ax.imshow(values, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(table.shape[1]))
        ax.set_xticklabels(table.columns, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(table.shape[0]))
        ax.set_yticklabels(table.index)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                ax.text(j, i, fmt % values[i, j], ha="center", va="center", fontsize=9)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Portfolio grid: strategy menu (rows) x weighting method (columns)")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_portfolio_grid.png", dpi=150)
    plt.close(fig)

    weights = relabel(results[headline_name]["weights_capped"])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.stackplot(weights.index, [weights[c].values for c in weights.columns],
                 labels=list(weights.columns), alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_title("Core3 inverse-vol portfolio: strategy weights over time (after the turnover cap)")
    ax.legend(ncol=4, fontsize=9, loc="upper left")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_portfolio_weights.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    for ax, (table, title) in zip(axes, ((sub_strategies, "Strategies"), (sub_ports, "Portfolios"))):
        x = np.arange(table.shape[1])
        width = 0.8 / table.shape[0]
        for k, (name, row) in enumerate(table.iterrows()):
            ax.bar(x + k * width, row.values.astype(float), width, label=name, color=COLORS.get(name))
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels(table.columns)
        ax.set_ylabel("Net Sharpe")
        ax.set_title(f"Sub-period Sharpe ratios: {title}")
        ax.legend(ncol=4, fontsize=8)
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_subperiod_sharpe.png", dpi=150)
    plt.close(fig)

    levels = [float(c[:-3]) for c in cost_strategies.columns
              if c.endswith("bps") and c[:-3].replace(".", "").isdigit()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for name in cost_strategies.index:
        ax1.plot(levels, cost_strategies.loc[name, [f"{l:g}bps" for l in levels]].values.astype(float),
                 marker="o", label=name, color=COLORS.get(name))
    ax1.axhline(sharpe(spy.loc[oos_start:]), color="black", ls="--", lw=1, label="SPY buy & hold")
    ax1.set_xlabel("cost per unit of turnover (bps)")
    ax1.set_ylabel("net Sharpe")
    ax1.set_title("Cost sensitivity: strategies")
    ax1.legend(fontsize=8)
    for column in cost_ports.columns:
        ax2.plot(levels, cost_ports[column].values.astype(float), marker="o", label=column)
    ax2.axhline(sharpe(spy.loc[port_start:]), color="black", ls="--", lw=1, label="SPY buy & hold")
    ax2.set_xlabel("cost per unit of turnover (bps, strategy and portfolio level)")
    ax2.set_title("Cost sensitivity: inverse-vol portfolios")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_cost_sensitivity.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, (table, title) in zip(axes, ((grid_aaa, "AAA: top-N x momentum lookback"),
                                         (grid_aaa_vol, "AAA: vol window x defensive asset"))):
        values = table.values.astype(float)
        im = ax.imshow(values, cmap="RdYlGn", vmin=0, vmax=1.3, aspect="auto")
        ax.set_xticks(range(table.shape[1]))
        ax.set_xticklabels(table.columns, rotation=20, ha="right", fontsize=9)
        ax.set_yticks(range(table.shape[0]))
        ax.set_yticklabels(table.index, fontsize=9)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"AAA parameter sensitivity (net Sharpe, out-of-sample window; "
                 f"SPY buy & hold = {sharpe(spy.loc[oos_start:]):.2f})")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_param_sensitivity.png", dpi=150)
    plt.close(fig)

    sharpe_columns = [c for c in overlay.columns if c.endswith("Sharpe")]
    dd_columns = [c for c in overlay.columns if c.endswith("MaxDD")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for column in sharpe_columns:
        ax1.plot(overlay.index, overlay[column].values.astype(float), marker="o",
                 label=column.replace(" Sharpe", ""))
    ax1.set_title("Core3 inv-vol: Sharpe vs turnover cap and vol target")
    ax1.set_ylabel("net Sharpe")
    ax1.legend(fontsize=8)
    for column in dd_columns:
        # Percent, to match the drawdown units used in every table in the report.
        ax2.plot(overlay.index, overlay[column].values.astype(float) * 100.0, marker="o",
                 label=column.replace(" MaxDD", ""))
    ax2.set_title("Core3 inv-vol: max drawdown vs turnover cap and vol target")
    ax2.set_ylabel("max drawdown (%)")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_overlay_sensitivity.png", dpi=150)
    plt.close(fig)

    short_names = {"2008 global financial crisis": "2008\nGFC",
                   "2010 flash crash / euro scare": "2010\nflash crash",
                   "2011 US downgrade / euro crisis": "2011\nUS downgrade",
                   "2015 China deval / Aug flash crash": "2015\nChina deval",
                   "2018 Feb VIX spike": "2018 Feb\nVIX spike",
                   "2018 Q4 rate-hike selloff": "2018 Q4\nselloff",
                   "2020 COVID crash": "2020\nCOVID",
                   "2022 inflation / hiking cycle": "2022\nhiking cycle",
                   "2025 tariff shock": "2025\ntariffs"}
    show = stress[["VolTiming", "RegimeHMM", "AAA", "Core3 inv-vol", "SPY", "60/40"]]
    fig, ax = plt.subplots(figsize=(12, 5.0))
    x = np.arange(show.shape[0])
    width = 0.8 / show.shape[1]
    for k, column in enumerate(show.columns):
        ax.bar(x + k * width, show[column].values.astype(float) * 100, width,
               label=column, color=COLORS.get(column))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels([short_names.get(e, e) for e in show.index], fontsize=8)
    ax.set_ylabel("total return in episode (%)")
    ax.set_title("Stress episodes: total return, walk-forward out of sample")
    ax.legend(ncol=3, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_stress_tests.png", dpi=150)
    plt.close(fig)

    paths = monte_carlo_paths_frame(headline.dropna(), n_paths=200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(paths, color="tab:blue", alpha=0.06, lw=0.8)
    ax1.plot(paths.median(axis=1), color="tab:blue", lw=2, label="median path")
    ax1.plot(paths.quantile(0.05, axis=1), color="tab:red", lw=1.5, ls="--", label="5th percentile")
    ax1.plot(paths.quantile(0.95, axis=1), color="tab:green", lw=1.5, ls="--", label="95th percentile")
    ax1.set_title("Core3 inv-vol: 1-year block-bootstrap Monte Carlo (200 of 5000 paths)")
    ax1.set_xlabel("trading days")
    ax1.set_ylabel("growth of $1")
    ax1.legend()
    style_axis(ax1)
    mc_names = ["Core3 inv-vol", "Orig2 inv-vol", "AAA", "SPY", "60/40"]
    ax2.bar(np.arange(len(mc_names)) - 0.2, mc.loc[mc_names, "p5_return"].astype(float) * 100,
            0.4, label="5th pct 1y return", color="tab:red", alpha=0.85)
    ax2.bar(np.arange(len(mc_names)) + 0.2, mc.loc[mc_names, "median_return"].astype(float) * 100,
            0.4, label="median 1y return", color="tab:blue", alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(range(len(mc_names)))
    ax2.set_xticklabels(mc_names, fontsize=8, rotation=15, ha="right")
    ax2.set_ylabel("%")
    ax2.set_title("1-year outcome distribution")
    ax2.legend(fontsize=8)
    style_axis(ax2)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_monte_carlo.png", dpi=150)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    ax1.plot(rolling_sharpe_ratio(headline, 252), label="Core3 inv-vol", lw=1.3)
    ax1.plot(rolling_sharpe_ratio(key_ports["Orig2 inv-vol"], 252), label="Orig2 inv-vol", lw=1.1, alpha=0.8)
    ax1.plot(rolling_sharpe_ratio(spy.loc[port_start:], 252), label="SPY", color="black", lw=1.2, alpha=0.6)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_title("Rolling 1-year Sharpe ratio")
    ax1.legend()
    ax2.plot(rolling_correlation(headline, spy, 252), label="Core3 inv-vol vs SPY", lw=1.2)
    ax2.plot(rolling_correlation(strategies["adaptive_allocation"], spy, 252),
             label="AAA vs SPY", lw=1.1, alpha=0.8, color="tab:green")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Rolling 1-year correlation to SPY")
    ax2.legend()
    for ax in (ax1, ax2):
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_rolling_sharpe.png", dpi=150)
    plt.close(fig)

    regime_sharpe = regime_conditioned["sharpe"].unstack("regime").reindex(
        columns=["bull", "sideways", "crisis"])
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(regime_sharpe.shape[0])
    for k, column in enumerate(regime_sharpe.columns):
        ax.bar(x + k * 0.26, regime_sharpe[column].values.astype(float), 0.26, label=f"{column} regime")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x + 0.26)
    ax.set_xticklabels(regime_sharpe.index, fontsize=9)
    ax.set_ylabel("Sharpe inside regime")
    ax.set_title("Regime-conditioned Sharpe (HMM labels on SPY, out-of-sample window)")
    ax.legend()
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_regime_conditioned.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.6))
    show = calendar_returns[["Core3 inv-vol", "Orig2 inv-vol", "SPY", "60/40"]]
    x = np.arange(show.shape[0])
    for k, column in enumerate(show.columns):
        ax.bar(x + k * 0.2, show[column].values.astype(float) * 100, 0.2,
               label=column, color=COLORS.get(column))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x + 0.3)
    ax.set_xticklabels(show.index, rotation=45)
    ax.set_ylabel("calendar-year return (%)")
    ax.set_title("Calendar-year returns")
    ax.legend()
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_calendar_year_returns.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 10. Daily return streams for further analysis
    # ------------------------------------------------------------------
    pd.concat([relabel(strategies), pd.DataFrame(portfolios), spy, sixty_forty],
              axis=1).to_csv(RESULTS / "daily_returns.csv")
    regime_labels.to_csv(RESULTS / "regime_labels.csv")
    print(f"\nDone in {time.time() - t0:.0f}s. Tables and figures in results/. "
          f"Run research_summary/make_summary_assets.py to refresh the write-ups.")


if __name__ == "__main__":
    main()
