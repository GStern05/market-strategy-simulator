import numpy as np
import pandas as pd
import pytest

from market_robustness.backtest.baseline import run_historical_baseline
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
from market_robustness.strategy.moving_average import MovingAverageStrategy


@pytest.fixture
def sample_strategy():
    return MovingAverageStrategy(short_window=8, long_window=20)


def test_run_historical_baseline_returns_expected_keys(sample_strategy):
    result = run_historical_baseline("AAPL", "2020-01-01", "2025-01-01", sample_strategy, 10000)
    expected_keys = {
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


def test_run_historical_baseline_values_correct_on_known_data(mocker, sample_strategy):
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    prices = pd.Series(
        [100 + i + (5 if i % 10 == 0 else 0) for i in range(30)],
        index=dates,
    )
    price_data = pd.DataFrame({"Close": prices})

    mocker.patch(
        "market_robustness.backtest.baseline.load_price_data",
        return_value=price_data,
    )

    result = run_historical_baseline("AAPL", "2024-01-01", "2024-01-30", sample_strategy, 10000)

    expected_backtest = run_backtest(price_data, sample_strategy, 10000)
    expected = {
        "cumulative_return": cumulative_return(expected_backtest["portfolio_value"], 10000),
        "final_portfolio_value": final_portfolio_value(expected_backtest["portfolio_value"]),
        "annualised_return": annualised_return(expected_backtest["portfolio_value"], 10000),
        "annualised_volatility": annualised_volatility(expected_backtest["portfolio_returns"]),
        "sharpe_ratio": sharpe_ratio(
            expected_backtest["portfolio_value"], 10000, expected_backtest["portfolio_returns"]
        ),
        "maximum_drawdown": maximum_drawdown(expected_backtest["portfolio_value"]),
        "number_of_trades": number_of_trades(expected_backtest["positions"]),
        "win_rate": win_rate(expected_backtest["positions"], expected_backtest["portfolio_returns"]),
    }

    for key in expected:
        assert result[key] == pytest.approx(expected[key], nan_ok=True)

        
def test_run_historical_baseline_calls_dependencies_correctly(mocker, sample_strategy):
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    prices = pd.Series(range(100, 110), index=dates)
    price_data = pd.DataFrame({"Close": prices})

    mock_load = mocker.patch(
        "market_robustness.backtest.baseline.load_price_data",
        return_value=price_data,
    )
    mock_run_backtest = mocker.patch(
        "market_robustness.backtest.baseline.run_backtest",
        return_value={
            "positions": pd.Series([0] * 10, index=dates),
            "portfolio_returns": pd.Series([0.0] * 10, index=dates),
            "portfolio_value": pd.Series([10000.0] * 10, index=dates),
        },
    )

    run_historical_baseline("AAPL", "2024-01-01", "2024-01-10", sample_strategy, 10000)

    mock_load.assert_called_once_with("AAPL", "2024-01-01", "2024-01-10")
    mock_run_backtest.assert_called_once_with(price_data, sample_strategy, 10000)


def test_run_historical_baseline_real_data_integration(sample_strategy):
    result = run_historical_baseline("AAPL", "2020-01-01", "2025-01-01", sample_strategy, 10000)

    assert isinstance(result["cumulative_return"], float)
    assert isinstance(result["sharpe_ratio"], float)
    assert isinstance(result["number_of_trades"], (int, np.integer))
    assert not np.isnan(result["cumulative_return"])
    assert not np.isnan(result["maximum_drawdown"])