import pandas as pd


def calculate_returns(df: pd.DataFrame) -> pd.Series:
    returns = df["Close"].pct_change()
    return returns