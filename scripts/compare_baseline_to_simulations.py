import json

import matplotlib.pyplot as plt

from market_robustness.backtest.baseline import run_historical_baseline
from market_robustness.data.loader import load_price_data
from market_robustness.data.returns import calculate_returns
from market_robustness.simulation.analysis import (
    compare_historical_to_simulated,
    simulations_to_dataframe,
)
from market_robustness.simulation.monte_carlo import run_monte_carlo_simulation
from market_robustness.strategy.moving_average import MovingAverageStrategy

strategy = MovingAverageStrategy(short_window=20, long_window=50)
initial_capital = 10000
historical_baseline = run_historical_baseline("AAPL", "2020-01-01", "2025-01-01", strategy, initial_capital)

price_data = load_price_data("AAPL", "2020-01-01", "2025-01-01")
stock_returns = calculate_returns(price_data)

n_simulations = 100
block_size = 10
simulations = run_monte_carlo_simulation(stock_returns, block_size, n_simulations, strategy, initial_capital)

df = simulations_to_dataframe(simulations)

result = compare_historical_to_simulated(historical_baseline, df)
print(json.dumps(result, indent=2, default=float))

# --- Histogram ---
plt.figure(figsize=(12, 5))
plt.hist(df["cumulative_return"], bins=20, color="blue", label="Simulated returns")
plt.axvline(
    historical_baseline["cumulative_return"],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Historical return",
)
plt.title("AAPL Cumulative Return: Historical vs. Simulated Distribution")
plt.xlabel("Cumulative Return")
plt.ylabel("Number of Simulations")
plt.legend()
plt.tight_layout()
plt.savefig("scripts/cumulative_return_historical_vs_simulated.png")
plt.close()