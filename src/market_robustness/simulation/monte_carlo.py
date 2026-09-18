from typing import Any

import numpy as np
import pandas as pd

from market_robustness.backtest.engine import run_backtest
from market_robustness.metrics.metrics import (
    annualised_return,
    annualised_volatility,
    cumulative_return,
    final_portfolio_value,
    maximum_drawdown,
    number_of_trades,
    sharpe_ratio,
    win_rate,
)
from market_robustness.simulation.bootstrap import block_bootstrap


def returns_to_prices(returns: pd.Series, initial_price: float = 100.0) -> pd.DataFrame:
    price_series = initial_price * (1 + returns).cumprod()
    return pd.DataFrame({"Close": price_series})


def run_single_simulation(
    returns: pd.Series,
    block_size: int,
    seed: int,
    simulation_id: int,
    strategy: Any,
    initial_capital: float,
) -> dict[str, Any]:
    simulation_returns = block_bootstrap(returns, block_size, seed)
    simulation = returns_to_prices(simulation_returns)
    backtest = run_backtest(simulation, strategy, initial_capital)

    return {
        "simulation_id": simulation_id,
        "seed": seed,
        "cumulative_return": cumulative_return(backtest["portfolio_value"], initial_capital),
        "final_portfolio_value": final_portfolio_value(backtest["portfolio_value"]),
        "annualised_return": annualised_return(backtest["portfolio_value"], initial_capital),
        "annualised_volatility": annualised_volatility(backtest["portfolio_returns"]),
        "sharpe_ratio": sharpe_ratio(
            backtest["portfolio_value"], initial_capital, backtest["portfolio_returns"]
        ),
        "maximum_drawdown": maximum_drawdown(backtest["portfolio_value"]),
        "number_of_trades": number_of_trades(backtest["positions"]),
        "win_rate": win_rate(backtest["positions"], backtest["portfolio_returns"]),
    }


def run_monte_carlo_simulation(
    returns: pd.Series,
    block_size: int,
    n_simulations: int,
    strategy: Any,
    initial_capital: float,
) -> list[dict[str, Any]]:
    simulations: list[dict[str, Any]] = []
    degenerate_count = 0

    for i in range(n_simulations):
        result = run_single_simulation(returns, block_size, i, i, strategy, initial_capital)
        if np.isnan(result["sharpe_ratio"]) or np.isnan(result["win_rate"]):
            degenerate_count += 1
        simulations.append(result)

    if degenerate_count > 0:
        print(
            f"Warning: {degenerate_count} out of {n_simulations} simulations "
            f"produced degenerate metrics (nan sharpe_ratio or win_rate)."
        )

    return simulations