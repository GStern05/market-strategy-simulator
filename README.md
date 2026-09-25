# Market Strategy Robustness Simulator

A Python tool for evaluating how robust a trading strategy really is, by backtesting it across thousands of statistically generated versions of market history rather than just one.

## Overview

A conventional backtest only tests a trading strategy against one historical sequence of events. If a strategy performs well over that period, it is difficult to know whether the result reflects a genuinely robust strategy or whether it benefited from the particular order in which market events happened.

This project takes historical market data and generates thousands of alternative but statistically realistic market histories using a block-bootstrap approach. Resampling blocks of consecutive days, rather than individual days, preserves short-term structure in the data such as volatility clustering. The same trading strategy is then backtested against each simulated history, producing a distribution of returns, drawdowns and other performance measures rather than a single historical result.

## Tech stack

Python, pandas, NumPy, pytest (add/adjust to match requirements.txt)

## Setup

Requires Python 3.X+. Clone the repository and set up the environment:

```bash
git clone https://github.com/GStern05/market-strategy-simulator.git
cd market-strategy-simulator
python3 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running tests

```bash
pytest -v
```

## Roadmap

- [x] Development environment (Git, virtual environment, dependency management, testing, linting, type checking)
- [x] Data loading: fetches historical price data, validates it (rejects invalid tickers, missing values and non-chronological data), and calculates daily returns, covered by tests using mocked data
- [x] Block-bootstrap generation of simulated market histories
- [x] Trading strategy interface and example strategies
- [x] Backtesting engine and performance metrics (returns, drawdowns, Sharpe ratio)
- [ ] Parallelising simulations for performance
- [ ] Results visualisation