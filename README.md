# Systematic Strategy Research Pipeline

A four-stage quantitative research pipeline on daily ETF data (2004 – August 2026):

1. **Signal discovery** – forecast SPY volatility (GARCH, random forest) and identify market regimes (Hidden Markov Model).
2. **Strategy construction** – turn the signals into trading rules and validate them walk-forward, after costs.
3. **Portfolio construction** – combine the strategies with equal, inverse-volatility and Markowitz weights, under a turnover cap, transaction costs and a volatility target.
4. **Robustness and risk** – sub-period, cost, parameter and overlay sensitivity, historical stress episodes and a block-bootstrap Monte Carlo.

**All strategy and portfolio results are walk-forward out of sample** (July 2006 – August 2026): models are refit on expanding training windows and rule parameters are chosen on training data only.

The write-ups are in [`research_summary/`](research_summary/):
[full report](research_summary/full_QR_summary.pdf) and [short summary](research_summary/QR_short_summary.pdf).

## Data

Adjusted daily close prices for eight ETFs from `yfinance`: SPY, TLT, GLD, USO, UUP, EFA, EEM and FXE (U.S. equities, Treasuries, gold, oil, the dollar, developed and emerging-market equities, the euro). ETFs that started trading after 2004 enter the data on their first trading day.

## Strategies

| Strategy | Rule |
|---|---|
| **VolTiming** | SPY exposure proportional to 1 / forecast volatility (random forest), capped at 1.5× |
| **RegimeHMM** | 3-state HMM on SPY; 100% / 35% / 0% SPY in bull / sideways / crisis states |
| **AAA** (adaptive asset allocation) | Monthly: rank the ETFs by past return, keep the strongest, weight by inverse volatility, move any holding with negative momentum to TLT |

## Key findings

- Adding AAA to the two SPY-based strategies raised the portfolio Sharpe ratio under every weighting method, mainly through diversification.
- Simple weighting (equal and inverse volatility) did at least as well as Markowitz optimization.
- The portfolios gave up some growth relative to SPY in exchange for much smaller drawdowns; the weak periods were 2022, when stocks and bonds fell together, and short, sudden shocks.

These findings come from one historical sample, not a forecast of future performance. The full results, methods and limitations are in the [full report](research_summary/full_QR_summary.pdf).

## Repository layout

```text
notebooks/
  01_signal_and_model_discovery.ipynb   # volatility models and HMM regimes
  02_strategy_construction.ipynb        # strategies and walk-forward results
  03_portfolio_and_execution.ipynb      # portfolios, robustness and risk
src/
  utils/        # configuration, data download and caching, features
  models/       # volatility (RF, GARCH), HMM regimes, and exploratory spread models
  strategies/   # strategy rules and benchmarks
  backtest/     # backtest engine and walk-forward validation
  portfolio/    # equal, inverse-vol and Markowitz weighting
  execution/    # turnover, costs, turnover cap
  risk/         # stress episodes, block-bootstrap Monte Carlo, drawdowns
  analysis/     # robustness checks and reporting helpers
run_research.py # runs the full pipeline
research_summary/  # full report and short summary (PDF)
```

The code also contains an exploratory pairs-trading rule; it is not part of the reported strategies or portfolios.

## How to run

```bash
pip install -r requirements.txt
python run_research.py
```

The first run downloads prices into `data/cache/` and writes tables and figures to `results/` (both created locally). It takes about ten minutes, mostly the random-forest refits; later runs can reuse cached walk-forward positions with:

```bash
python run_research.py --fast
```

## Research principles

- **No look-ahead:** positions are lagged one day, rules use only past data, and models and parameters are fitted on training windows only.
- **Costs included:** 7 bps per unit of turnover at the strategy level and again at the portfolio level, with cost grids up to 30 bps.
- **Out of sample only:** every reported number comes from the walk-forward.
- **Robustness over point estimates:** sub-periods, parameter and cost grids, stress episodes and Monte Carlo.
