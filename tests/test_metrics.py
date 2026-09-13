import numpy as np
import pandas as pd
import pytest

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


def test_cumulative_return():
    portfolio_value = pd.Series([10000, 10500, 11000])
    result = cumulative_return(portfolio_value, initial_capital=10000)
    assert result == pytest.approx(0.10)


def test_final_portfolio_value():
    portfolio_value = pd.Series([10000, 10500, 11000])
    result = final_portfolio_value(portfolio_value)
    assert result == pytest.approx(11000)


def test_annualised_return():
    # 10% total return over 2 trading days -> hugely amplified once annualised.
    portfolio_value = pd.Series([10000, 11000])
    result = annualised_return(portfolio_value, initial_capital=10000)
    expected = (1.10) ** (252 / 2) - 1
    assert result == pytest.approx(expected)


def test_annualised_volatility():
    portfolio_returns = pd.Series([0.01, -0.01, 0.02, -0.02])
    result = annualised_volatility(portfolio_returns)
    expected = portfolio_returns.std() * np.sqrt(252)
    assert result == pytest.approx(expected)


def test_annualised_volatility_zero_when_returns_constant():
    # Edge case: no variation in returns at all -> volatility should be exactly 0.
    portfolio_returns = pd.Series([0.0, 0.0, 0.0, 0.0])
    result = annualised_volatility(portfolio_returns)
    assert result == pytest.approx(0.0)


def test_sharpe_ratio():
    portfolio_value = pd.Series([10000, 10100, 10200, 10150])
    portfolio_returns = portfolio_value.pct_change().fillna(0)
    result = sharpe_ratio(portfolio_value, 10000, portfolio_returns)
    expected = annualised_return(portfolio_value, 10000) / annualised_volatility(
        portfolio_returns
    )
    assert result == pytest.approx(expected)


def test_sharpe_ratio_is_nan_when_volatility_is_zero():
    # Documents current behaviour on a degenerate input, rather than asserting
    # it's "correct" -- flagged as a design decision to revisit.
    portfolio_value = pd.Series([10000, 10000, 10000])
    portfolio_returns = pd.Series([0.0, 0.0])
    result = sharpe_ratio(portfolio_value, 10000, portfolio_returns)
    assert np.isnan(result)


def test_maximum_drawdown():
    portfolio_value = pd.Series([10000, 11000, 10500, 9000, 9500, 12000, 10800])
    result = maximum_drawdown(portfolio_value)
    assert result == pytest.approx((9000 - 11000) / 11000)


def test_maximum_drawdown_zero_when_only_rising():
    portfolio_value = pd.Series([10000, 10500, 11000, 11500])
    result = maximum_drawdown(portfolio_value)
    assert result == pytest.approx(0.0)


def test_number_of_trades():
    positions = pd.Series([0, 1, 1, 1, -1, -1, 0, 0])
    result = number_of_trades(positions)
    assert result == 2


def test_number_of_trades_is_zero_when_never_in_market():
    positions = pd.Series([0, 0, 0, 0])
    result = number_of_trades(positions)
    assert result == 0


def test_win_rate():
    positions = pd.Series([0, 1, 1, 1, -1, -1, 0, 0])
    portfolio_returns = pd.Series([0, 0.01, 0.02, -0.01, 0.03, -0.02, 0, 0])
    # Trade 1 (days 1-3): (1.01 * 1.02 * 0.99) - 1 ~= +0.0188 -> win
    # Trade 2 (days 4-5): (1.03 * 0.98) - 1 ~= +0.0094 -> win
    result = win_rate(positions, portfolio_returns)
    assert result == pytest.approx(1.0)


def test_win_rate_with_a_losing_trade():
    positions = pd.Series([0, 1, 1, -1, -1, 0])
    portfolio_returns = pd.Series([0, 0.05, 0.05, -0.10, -0.10, 0])
    # Trade 1 (days 1-2): (1.05 * 1.05) - 1 -> win
    # Trade 2 (days 3-4): (0.90 * 0.90) - 1 -> loss
    result = win_rate(positions, portfolio_returns)
    assert result == pytest.approx(0.5)


def test_win_rate_is_nan_when_no_trades():
    positions = pd.Series([0, 0, 0, 0])
    portfolio_returns = pd.Series([0.0, 0.0, 0.0, 0.0])
    result = win_rate(positions, portfolio_returns)
    assert np.isnan(result)