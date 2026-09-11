import numpy as np
import pandas as pd

from market_robustness.strategy.base import Strategy


class MovingAverageStrategy(Strategy):
    """
    A trading strategy based on a moving-average crossover.

    Generates a buy signal when the short-term moving average is
    above the long-term moving average, and a sell signal when it
    is below. Implements the Strategy interface.
    """

    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        """
        Generate signals from a short/long moving-average crossover.

        Args:
            price_data: DataFrame of OHLCV price data, indexed by date.

        Returns:
            A pd.Series, indexed by the same dates as price_data, where
            each value is 1 (long/buy), -1 (short/sell), or 0 (hold/flat,
            used when there isn't yet enough data to compute both moving
            averages).
        """
        short_ma = price_data["Close"].rolling(window=self.short_window).mean()
        long_ma = price_data["Close"].rolling(window=self.long_window).mean()
        result = np.where(long_ma.isna(), 0, np.where(short_ma > long_ma, 1, -1))
        return pd.Series(result, index=price_data.index)