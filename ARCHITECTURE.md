# Architecture

## Dependency direction

data/      <- depended on by strategy/, backtest/

strategy/  <- depended on by backtest/, simulation/

backtest/  <- depended on by simulation/ (called by both the historical path and the simulation path)

metrics/   <- depended on by backtest/, simulation/

simulation/ <- depended on by parallel/, cli/

parallel/  <- depended on by cli/

cli/       <- depends on everything above; nothing depends on cli/

## Pipelines

```
Market Data
     ↓
Strategy
     ↓
Backtesting Engine
     ↓
Performance Metrics
```
```
Historical Returns
     ↓
Alternative History Generator
     ↓
Monte Carlo Simulation
     ↓
Backtesting Engine
     ↓
Simulation Results
```

## data/

Corresponds to "Market Data"/"Historical Returns". Loads stock prices, cleans and calculates returns.

## strategy/

Given price data, decides buy/hold/sell signals. 

## backtest/

Takes price data + strategy's signals, and simulates holding positions and trading, to produce a portfolio value over time.

## metrics/

Pure calculations on returns/equity curves — Sharpe ratio, volatility, maximum drawdown, etc. Doesn't know about strategies, backtesting mechanics, or simulations — just takes numbers in, gives numbers out. Reused in three places: once on the historical baseline, once per simulation, and again when summarising the distribution of thousands of results

## simulation/

Take real historical returns, generate a fake-but-realistic alternative history (block bootstrap), and orchestrate running that alternative history through the backtest engine, thousands of times.

## parallel/

Takes the 'run N simulations' workload and spreads it across CPU cores instead of running them one at a time.  

The parallelisation layer should sit around the independent simulations:
```
                Simulation Engine
                      
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
     Worker 1       Worker 2       Worker 3 ...
        ↓              ↓              ↓
     Results        Results        Results
        └──────────────┼──────────────┘
                       ↓
                 Final Analysis
```
## cli/

Parses command-line arguments and calls the other modules. 
