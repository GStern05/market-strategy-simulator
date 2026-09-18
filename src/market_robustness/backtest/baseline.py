from typing import Any

from market_robustness.backtest.engine import run_backtest
from market_robustness.data.loader import load_price_data
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


def run_historical_baseline(
    ticker: str, start: str, end: str, strategy, initial_capital: float
) -> dict[str, Any]:
    price_data = load_price_data(ticker, start, end)
    backtest = run_backtest(price_data, strategy, initial_capital)

    return {
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