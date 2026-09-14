from market_robustness.backtest.engine import run_backtest
from market_robustness.data.loader import load_price_data
from market_robustness.data.returns import calculate_returns
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

import matplotlib.pyplot as plt

positions = backtest['positions']
changed = positions != positions.shift(1).fillna(0)

buy_dates = positions.index[changed & (positions == 1)]
sell_dates = positions.index[changed & (positions == -1)]
flat_dates = positions.index[changed & (positions == 0)]

# --- Price chart with buy/sell markers ---
plt.figure(figsize=(12, 5))
plt.plot(price_data.index, price_data['Close'], label='AAPL Close', color='black', linewidth=1)
plt.scatter(buy_dates, price_data.loc[buy_dates, 'Close'], marker='^', color='green', label='Buy/Long', zorder=5)
plt.scatter(sell_dates, price_data.loc[sell_dates, 'Close'], marker='v', color='red', label='Sell/Short', zorder=5)
plt.scatter(flat_dates, price_data.loc[flat_dates, 'Close'], marker='x', color='gray', label='Exit to flat', zorder=5)
plt.title('AAPL Price with Strategy Signals')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.tight_layout()
plt.savefig('scripts/baseline_price_chart.png')
plt.close()

# --- Equity curve ---
plt.figure(figsize=(12, 5))
plt.plot(backtest['portfolio_value'].index, backtest['portfolio_value'], color='blue')
plt.title('Portfolio Equity Curve')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.tight_layout()
plt.savefig('scripts/baseline_equity_curve.png')
plt.close()