import pandas as pd

from market_robustness.data.returns import calculate_returns


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
    return positions * stock_returns.fillna(0)


def calculate_portfolio_value(portfolio_returns: pd.Series, initial_capital: float) -> pd.Series:
    """
    Convert daily portfolio returns into a portfolio value curve.
    """
    portfolio_value = initial_capital*(1+portfolio_returns).cumprod()
    return portfolio_value

def run_backtest(price_data: pd.DataFrame, strategy, initial_capital: float = 10000) -> dict:
    """
    Run a full backtest: signals -> positions -> portfolio returns -> portfolio value.
    """
    signals = strategy.generate_signals(price_data)
    positions = signals_to_positions(signals)
    stock_returns = calculate_returns(price_data)
    portfolio_returns = calculate_portfolio_returns(positions, stock_returns)
    portfolio_value = calculate_portfolio_value(portfolio_returns, initial_capital)

    return {
        "positions": positions,
        "portfolio_returns": portfolio_returns,
        "portfolio_value": portfolio_value,
    }