from market_robustness.data.loader import load_price_data
from market_robustness.data.returns import calculate_returns
from market_robustness.strategy.moving_average import MovingAverageStrategy
from market_robustness.backtest.engine import run_backtest
from market_robustness.metrics.metrics import cumulative_return, final_portfolio_value, annualised_return, annualised_volatility, sharpe_ratio, maximum_drawdown, number_of_trades, win_rate

## print every metric

price_data = load_price_data("AAPL","2020-01-01","2025-01-01")
stock_returns = calculate_returns(price_data)

strategy = MovingAverageStrategy(short_window =20, long_window =50)
initial_capital = 100

backtest = run_backtest(price_data,strategy,initial_capital)

print(f"Cumulative Return: {cumulative_return(backtest['portfolio_value'], initial_capital)}")
print(f"Final Portfolio Value: {final_portfolio_value(backtest['portfolio_value'])}")
print(f"Annualised Return: {annualised_return(backtest['portfolio_value'], initial_capital)}")
print(f"Annualised Volatility: {annualised_volatility(backtest['portfolio_returns'])}")
print(f"Sharpe Ratio: {sharpe_ratio(backtest['portfolio_value'], initial_capital, backtest['portfolio_returns'])}")
print(f"Maximum Drawdown: {maximum_drawdown(backtest['portfolio_value'])}")
print(f"Number of Trades: {number_of_trades(backtest['positions'])}")
print(f"Win Rate: {win_rate(backtest['positions'], backtest['portfolio_returns'])}")

