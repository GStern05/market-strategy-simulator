from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    '''
    The interface all trading strategies must implement.
    '''
    @abstractmethod
    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from historical price data.

        Args:
            price_data: DataFrame of OHLCV price data, indexed by date.

        Returns:
            A pd.Series, indexed by the same dates as price_data, where
            each value is 1 (long/buy), 0 (hold/flat), or -1 (short/sell).
        """
