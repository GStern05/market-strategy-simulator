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

def percentile_rank(df: pd.DataFrame, historical_value: float, column: str = "cumulative_return") -> float:
    """Calculates the percentile rank of the historical value: the fraction
    of simulations that performed the same as or worse than it."""
    return 1 - probability_of_exceeding_historical(df, historical_value, column)

def classify_performance(rank: float) -> str:
    if rank <= 0.05:
        return "unusually poor"
    elif rank <= 0.25:
        return "poor"
    elif rank <= 0.75:
        return "typical"
    elif rank <= 0.95:
        return "good"
    else:
        return "unusually good"



def compare_historical_to_simulated(historical: dict, df: pd.DataFrame) -> dict:
    metrics = [
        "cumulative_return",
        "final_portfolio_value",
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "maximum_drawdown",
        "number_of_trades",
        "win_rate",
    ]

    result: dict[str, Any] = {}
    for metric in metrics:
        rank = percentile_rank(df, historical[metric], metric)
        result[metric] = {
            "historical_value": historical[metric],
            "percentile_rank": rank,
            "classification": classify_performance(rank),
        }

    return result
