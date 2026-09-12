import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from market_robustness.backtest.engine import (
    calculate_portfolio_returns,
    calculate_portfolio_value,
    run_backtest,
    signals_to_positions,
)
from market_robustness.data.loader import load_price_data
from market_robustness.strategy.moving_average import MovingAverageStrategy


def test_signals_to_positions():
    signals = pd.Series([0, 1, 1, -1, -1, 0])
    positions = signals_to_positions(signals)
    expected = pd.Series([0.0, 0.0, 1.0, 1.0, -1.0, -1.0])
    assert_series_equal(positions, expected)

def test_calculate_portfolio_returns():
    positions = pd.Series([0,0,1,1,-1,-1])
    stock_returns = pd.Series([0.00, 0.02, -0.01, 0.015, -0.03, 0.01])
    portfolio_return = calculate_portfolio_returns(positions,stock_returns)
    expected = pd.Series([0.000, 0.000, -0.010, 0.015, 0.030, -0.010])
    assert_series_equal(portfolio_return,expected)

def test_calculate_portfolio_value():
    portfolio_returns = pd.Series([0.000, 0.000, -0.010, 0.015, 0.030, -0.010])
    initial_capital = 10000
    portfolio_value = calculate_portfolio_value(portfolio_returns,initial_capital)
    expected = pd.Series([10000.0, 10000.0, 9900.0, 10048.5, 10349.955, 10246.45545])
    assert_series_equal(portfolio_value,expected)

def test_run_backtest():
    price_data = load_price_data("AAPL", "2023-01-01", "2023-06-01")
    strategy = MovingAverageStrategy(short_window=10, long_window=30)

    result = run_backtest(price_data, strategy, initial_capital=10000)

    assert isinstance(result["portfolio_value"], pd.Series)
    assert len(result["portfolio_value"]) == len(price_data)
    assert result["portfolio_value"].isna().sum() == 0
    assert result["portfolio_value"].iloc[0] == pytest.approx(10000, rel=0.05)