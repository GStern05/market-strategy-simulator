import numpy as np
import pandas as pd


def cumulative_return(portfolio_value: pd.Series, initial_capital: float) -> float:
    final_value = portfolio_value.iloc[-1]
    return (final_value / initial_capital) - 1


def final_portfolio_value(portfolio_value: pd.Series) -> float:
    return portfolio_value.iloc[-1]

def annualised_return(portfolio_value: pd.Series, initial_capital: float) -> float:
    total_return = cumulative_return(portfolio_value, initial_capital)
    return (1 + total_return) ** (252 / len(portfolio_value)) - 1

def annualised_volatility(portfolio_returns: pd.Series) -> float:
    return portfolio_returns.std() * np.sqrt(252)

def sharpe_ratio(portfolio_value: pd.Series, initial_capital: float, portfolio_returns: pd.Series)->float:
    annual_return = annualised_return(portfolio_value,initial_capital)
    annual_volatility = annualised_volatility(portfolio_returns)
    return annual_return/annual_volatility

def maximum_drawdown(portfolio_value: pd.Series)-> float:
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max
    return drawdown.min()

def number_of_trades(positions: pd.Series) -> int:
    changed = positions != positions.shift(1).fillna(0)
    trade_starts = changed & (positions != 0)
    return int(trade_starts.sum())

def win_rate(positions: pd.Series, portfolio_returns: pd.Series) -> float:
    changed = positions != positions.shift(1).fillna(0)
    trade_starts = changed & (positions != 0)
    trade_ids = trade_starts.cumsum()

    in_trade = positions != 0
    trade_returns = (
        portfolio_returns[in_trade]
        .groupby(trade_ids[in_trade])
        .apply(lambda g: (1 + g).prod() - 1) # type: ignore[operator]  # pandas .apply() return type is untyped; always float here
    )

    return (trade_returns > 0).mean()

