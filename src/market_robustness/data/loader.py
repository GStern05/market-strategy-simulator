import yfinance as yf
import pandas as pd


def load_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")
    df.columns = df.columns.droplevel('Ticker')
    if df.isna().sum().sum() != 0:
        raise ValueError(f"Missing values found in data for ticker '{ticker}'")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"Data for ticker '{ticker}' is not in chronological order")
    return df