import numpy as np
import pandas as pd
import pytest

from market_robustness.simulation.monte_carlo import (
    returns_to_prices,
    run_monte_carlo_simulation,
    run_single_simulation,
)
from market_robustness.strategy.moving_average import MovingAverageStrategy


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    return pd.Series(np.random.normal(loc=0.0005, scale=0.01, size=200))


@pytest.fixture
def sample_strategy():
    return MovingAverageStrategy(short_window=8, long_window=20)


def test_returns_to_prices_returns_dataframe_with_close_column(sample_returns):
    prices = returns_to_prices(sample_returns)
    assert isinstance(prices, pd.DataFrame)
    assert "Close" in prices.columns


def test_returns_to_prices_starts_at_initial_price(sample_returns):
    prices = returns_to_prices(sample_returns, initial_price=100.0)
    assert prices["Close"].iloc[0] == pytest.approx(100.0 * (1 + sample_returns.iloc[0]))


def test_run_single_simulation_returns_expected_keys(sample_returns, sample_strategy):
    result = run_single_simulation(
        sample_returns,
        block_size=20,
        seed=1,
        simulation_id=0,
        strategy=sample_strategy,
        initial_capital=10000,
    )
    expected_keys = {
        "simulation_id",
        "seed",
        "cumulative_return",
        "final_portfolio_value",
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "number_of_trades",
        "win_rate",
    }
    assert set(result.keys()) == expected_keys


def test_run_single_simulation_records_correct_id_and_seed(sample_returns, sample_strategy):
    result = run_single_simulation(
        sample_returns,
        block_size=20,
        seed=7,
        simulation_id=3,
        strategy=sample_strategy,
        initial_capital=10000,
    )
    assert result["simulation_id"] == 3
    assert result["seed"] == 7


def test_run_monte_carlo_simulation_returns_correct_number_of_results(
    sample_returns, sample_strategy
):
    results = run_monte_carlo_simulation(
        sample_returns,
        block_size=20,
        n_simulations=10,
        strategy=sample_strategy,
        initial_capital=10000,
    )
    assert len(results) == 10


def test_run_monte_carlo_simulation_uses_distinct_seeds(sample_returns, sample_strategy):
    results = run_monte_carlo_simulation(
        sample_returns,
        block_size=20,
        n_simulations=10,
        strategy=sample_strategy,
        initial_capital=10000,
    )
    seeds = [r["seed"] for r in results]
    assert seeds == list(range(10))


def test_run_monte_carlo_simulation_produces_varying_results(sample_returns, sample_strategy):
    results = run_monte_carlo_simulation(
        sample_returns,
        block_size=20,
        n_simulations=10,
        strategy=sample_strategy,
        initial_capital=10000,
    )
    cumulative_returns = [r["cumulative_return"] for r in results]
    assert len(set(cumulative_returns)) > 1