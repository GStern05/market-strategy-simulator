import yfinance as yf
import pandas as pd


def load_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")
    df.columns = df.columns.droplevel('Ticker')
    return df