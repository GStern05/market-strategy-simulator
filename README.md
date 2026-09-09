# Market Strategy Robustness Simulator

This is a software application for evaluating the robustness of trading strategies across many statistically generated versions of market history.

## Overview

The motivation is that a conventional backtest only tests a trading strategy against one historical sequence of events. If a strategy performs well over that period, it is difficult to know whether the result reflects a genuinely robust strategy or whether it benefited from the particular order in which market events happened.

My system will therefore take historical market data and generate thousands of alternative but statistically realistic market histories using a block-bootstrap approach. The same trading strategy will then be backtested against each simulated history, producing a distribution of possible returns, drawdowns and other performance measures rather than a single historical result.

## Setup

Clone the repository and set up the environment:

​```
git clone https://github.com/GStern05/market-strategy-simulator.git
cd market-strategy-simulator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
​``` 

## Running Tests

```
pytest -v

```

## Status

The development environment (Git, virtual environment, dependency
management, testing, linting, and type checking) is set up. The
data-loading component is implemented: it fetches historical price
data, validates it (rejecting invalid tickers, missing values, and
non-chronological data), and calculates daily returns. This is
covered by a test suite using mocked data. No trading strategy or
backtesting logic has been implemented yet.

