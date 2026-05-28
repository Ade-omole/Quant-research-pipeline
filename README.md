# Systematic Strategy Research Pipeline

A three-stage quantitative research pipeline that moves from signal discovery to tradable rules to portfolio construction and execution simulation.

1. **Mathematical modeling** — forecast volatility, test pair relationships, and infer market regimes.
2. **Strategy construction** — convert model outputs into position rules and backtest them with costs.
3. **Portfolio and execution** — combine strategies, apply risk budgets, and stress-test under realistic constraints.

The code is modular: notebooks stay readable, and reusable logic lives in `src/`.

## Core Topics

This project covers:

- working with real financial data from `yfinance`,
- building reusable research code instead of one-off notebook logic,
- comparing pure research metrics with trading metrics,
- understanding how transaction costs and turnover change the story,
- connecting stochastic models to actual portfolio decisions.

## Repository Layout

```text
data/
  cache/                 # locally cached adjusted-close prices (created on first run)
notebooks/
  01_signal_and_model_discovery.ipynb
  02_strategy_construction.ipynb
  03_portfolio_and_execution.ipynb
src/
  models/
  strategies/
  backtest/
  portfolio/
  execution/
  utils/
results/
requirements.txt
README.md
```

## Stage Overview

### Project 1: Signal and Model Discovery

Research-mode analysis on a fixed ETF universe (`SPY`, `TLT`, `GLD`, `USO`, `UUP`, `EFA`, `EEM`, `FXE`).

It studies:

- volatility forecasting with `GARCH(1,1)` and `RandomForest`,
- pairs research with cointegration and an Ornstein-Uhlenbeck spread view — including a **full pair scan** across the configured universe (not limited to one pair),
- market regime detection with a Hidden Markov Model.

The main outputs are research diagnostics rather than trading performance:

- forecast MSE and MAE,
- correlation between predicted and realized volatility,
- cointegration p-values across candidate pairs,
- OU mean-reversion estimates,
- regime summaries and regime persistence intuition.

### Project 2: Strategy Construction

This stage converts the research signals into tradable rules.

It builds:

- a volatility-timing strategy on `SPY`,
- a pairs-trading strategy — **`EFA`/`EEM` is the default in `config.py`**, but Notebook 1 screens other pairs in the universe; you can swap `pair_symbols` and rerun the same pipeline,
- a regime-switching allocation rule on `SPY`.

It also includes an optional **expanding walk-forward** block (`src/backtest/walk_forward.py`) that refits models on rolling training windows and stitches out-of-sample test positions, with a side-by-side comparison against full-sample static rules.

The main outputs are strategy metrics:

- Sharpe ratio,
- Sortino ratio,
- CAGR,
- max drawdown,
- Calmar ratio,
- historical VaR and expected shortfall (95%),
- turnover,
- trading-cost drag.

### Project 3: Portfolio and Execution Simulation

This stage treats the stage-two strategies as portfolio building blocks.

It adds:

- **Method A:** inverse-volatility strategy weighting,
- **Method B:** rolling long-only Markowitz max-Sharpe weighting,
- volatility targeting,
- turnover caps and execution-cost overlays,
- regime-conditioned stress testing,
- rolling Sharpe and correlation analysis.

The goal is to show how individual signals behave once they are combined, scaled, and traded under portfolio-level constraints.

## What To Read First

Recommended order:

1. `notebooks/01_signal_and_model_discovery.ipynb` for the research logic.
2. `notebooks/02_strategy_construction.ipynb` for converting signals into rules.
3. `notebooks/03_portfolio_and_execution.ipynb` for the portfolio and execution layer.

## Notebook Structure

The notebooks are organized for step-by-step execution:

- markdown cells explain why the next step exists,
- code cells are short and focused,
- variables use descriptive names,
- comments clarify non-obvious lines,
- repeated logic is moved into `src/` so the notebooks stay readable.

## Module Guide

- `src/utils/config.py`: shared settings for symbols, dates, cache behavior, and execution assumptions.
- `src/utils/data_loader.py`: download prices (cached locally after the first fetch), compute returns, realized volatility, and spreads.
- `src/models/volatility.py`: volatility forecast models and forecast evaluation.
- `src/models/ou.py`: cointegration testing, OU estimation (OLS-on-deltas or exact discrete OU / AR(1)), and spread z-scores.
- `src/models/regime.py`: HMM feature creation, estimation, and regime summaries.
- `src/strategies/`: convert model outputs into positions.
- `src/backtest/engine.py`: convert positions into returns, equity curves, drawdowns, and performance metrics (including VaR and ES).
- `src/backtest/walk_forward.py`: expanding-window walk-forward refit helpers for all three strategies.
- `src/portfolio/construction.py`: combine strategies, Markowitz optimization, and volatility targeting.
- `src/execution/costs.py`: turnover measurement, cost estimation, and turnover caps.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebooks and run them from top to bottom.

The first run downloads adjusted closes from Yahoo Finance into `data/cache/`. Later runs reuse that snapshot so metrics stay stable across kernel restarts. To pull fresh prices (for example after changing dates in `src/utils/config.py`), call `build_market_data_bundle(refresh=True)` or `download_adjusted_close(refresh=True)` in a notebook cell, or run `clear_price_cache()` from `src.utils.data_loader` and rerun.

## Model Comparison Framework

The repo is structured so you can compare ideas at three different levels:

1. **Research level:** Does the model explain or predict something meaningful?
2. **Strategy level:** Does that signal survive trading rules and costs?
3. **Portfolio level:** Does the strategy improve the total portfolio after diversification and execution constraints?

## Expected Trading Implications

The pipeline is set up to surface a few common quant-research patterns:

- better forecast accuracy does not automatically mean better trading performance,
- mean reversion can disappear when costs and turnover are introduced,
- regime models are most useful when they drive risk budgeting rather than perfect market timing,
- a diversified portfolio of imperfect strategies can be more robust than one elegant standalone signal.
