
import pandas as pd

from market_robustness.strategy.moving_average import MovingAverageStrategy


def test_moving_average_strategy():
    dates = pd.date_range(start="2024-01-01", periods=8, freq="D")
    prices = [10, 10, 10, 20, 30, 40, 50, 60]

    price_data = pd.DataFrame({"Close": prices}, index=dates)

    strategy = MovingAverageStrategy(short_window = 2, long_window=3)
    signals = strategy.generate_signals(price_data)

    assert len(signals) == len(price_data)
    assert signals.index.equals(price_data.index)
    assert signals.iloc[0] == 0
    assert signals.iloc[6] == 1
