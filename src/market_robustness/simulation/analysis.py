from typing import Any

import pandas as pd


def simulations_to_dataframe(simulations: list[dict[str, Any]]) -> pd.DataFrame:
     return pd.DataFrame(simulations)


def summary_statistics(df: pd.DataFrame, column: str) -> dict[str, float]:
    results: dict[str, float] = {}
    results["mean"] = df[column].mean()
    results["median"] = df[column].median()
    results["std"] = df[column].std()
    results["p5"] = df[column].quantile(0.05)
    results["p25"] = df[column].quantile(0.25)
    results["p75"] = df[column].quantile(0.75)
    results["p95"] = df[column].quantile(0.95)

    return results

def probability_of_exceeding_historical(df: pd.DataFrame,historical_value: float, column: str = "cumulative_return") -> float:
    return (df[column]> historical_value).mean()

def probability_of_loss(df: pd.DataFrame, column: str = "cumulative_return") -> float:
    return (df[column]<0).mean()

def analyse_distribution(df: pd.DataFrame, historical_return: float) -> dict[str, Any]:
     result: dict[str,Any] = {}
     result["return_stats"] = summary_statistics(df,"cumulative_return")
     result["drawdown_stats"] = summary_statistics(df,"maximum_drawdown")
     result["probability_of_loss"] = probability_of_loss(df)
     result["probability_exceeding_historical"] = probability_of_exceeding_historical(df, historical_return)
     return result
     