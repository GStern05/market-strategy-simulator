import pandas as pd

def signals_to_positions(signals: pd.Series) -> pd.Series:
    """
    Convert raw strategy signals into actual held positions,
    avoiding look-ahead bias.
    """
    positions = signals.shift(1).fillna(0)
    return positions

def calculate_portfolio_returns(positions: pd.Series, stock_returns: pd.Series) -> pd.Series:
    """
    Calculate daily portfolio returns from held positions and stock returns.
    """
    return positions * stock_returns


def calculate_portfolio_value(portfolio_returns: pd.Series, initial_capital: float) -> pd.Series:
    """
    Convert daily portfolio returns into a portfolio value curve.
    """
    portfolio_value = initial_capital*(1+portfolio_returns).cumprod()
    return portfolio_value