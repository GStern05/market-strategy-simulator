import pandas as pd
import pytest

from market_robustness.simulation.analysis import (
    analyse_distribution,
    classify_performance,
    compare_historical_to_simulated,
    percentile_rank,
    probability_of_exceeding_historical,
    probability_of_loss,
    simulations_to_dataframe,
    summary_statistics,
)


@pytest.fixture
def sample_simulations_df():
    return pd.DataFrame({
        "cumulative_return":     [-0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        "final_portfolio_value": [9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500],
        "annualised_return":     [-0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        "annualised_volatility": [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28],
        "sharpe_ratio":          [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3],
        "maximum_drawdown":      [-0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05],
        "number_of_trades":      [10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        "win_rate":              [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    })


def test_simulations_to_dataframe_converts_list_of_dicts():
    simulations = [
        {"cumulative_return": 0.05, "sharpe_ratio": 0.5},
        {"cumulative_return": -0.02, "sharpe_ratio": -0.1},
        {"cumulative_return": 0.10, "sharpe_ratio": 0.9},
    ]

    df = simulations_to_dataframe(simulations)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["cumulative_return", "sharpe_ratio"]
    assert len(df) == 3
    assert df["cumulative_return"].tolist() == [0.05, -0.02, 0.10]


def test_summary_statistics_returns_expected_keys_and_values(sample_simulations_df):
    result = summary_statistics(sample_simulations_df, "cumulative_return")
    expected_keys = {
        "mean",
        "median",
        "std",
        "p5",
        "p25",
        "p75",
        "p95",
    }
    assert set(result.keys()) == expected_keys

    assert result["mean"] == pytest.approx(0.125)
    assert result["median"] == pytest.approx(0.125)


def test_probability_of_loss(sample_simulations_df):
    result = probability_of_loss(sample_simulations_df)
    assert result == pytest.approx(0.2)

def test_probability_of_exceeding_historical(sample_simulations_df):
    result = probability_of_exceeding_historical(sample_simulations_df,0.1)
    assert result == pytest.approx(0.5)


def test_analyse_distribution_returns_expected_structure(sample_simulations_df):
    result = analyse_distribution(sample_simulations_df, 0.1)

    expected_keys = {
        "return_stats",
        "drawdown_stats",
        "probability_of_loss",
        "probability_exceeding_historical",
    }
    assert set(result.keys()) == expected_keys

    assert result["probability_of_loss"] == pytest.approx(0.2)
    assert result["probability_exceeding_historical"] == pytest.approx(0.5)
    assert result["return_stats"]["mean"] == pytest.approx(0.125)

def test_percentile_rank(sample_simulations_df):
    result = percentile_rank(sample_simulations_df, 0.0)
    assert result == pytest.approx(0.3)

def test_classify_performance_boundaries():
    assert classify_performance(0.0) == "unusually poor"
    assert classify_performance(0.05) == "unusually poor"
    assert classify_performance(0.06) == "poor"
    assert classify_performance(0.25) == "poor"
    assert classify_performance(0.26) == "typical"
    assert classify_performance(0.75) == "typical"
    assert classify_performance(0.76) == "good"
    assert classify_performance(0.95) == "good"
    assert classify_performance(0.96) == "unusually good"
    assert classify_performance(1.0) == "unusually good"


def test_compare_historical_to_simulated_returns_expected_structure(sample_simulations_df):
    historical = {
        "cumulative_return": 0.1,
        "final_portfolio_value": 11000,
        "annualised_return": 0.02,
        "annualised_volatility": 0.18,
        "sharpe_ratio": 0.3,
        "maximum_drawdown": -0.3,
        "number_of_trades": 18,
        "win_rate": 0.5,
    }

    result = compare_historical_to_simulated(historical, sample_simulations_df)

    expected_metrics = set(historical.keys())
    assert set(result.keys()) == expected_metrics

    for metric in expected_metrics:
        entry = result[metric]
        assert set(entry.keys()) == {"historical_value", "percentile_rank", "classification"}
        assert entry["historical_value"] == historical[metric]

    assert result["cumulative_return"]["percentile_rank"] == pytest.approx(0.5)
    assert result["cumulative_return"]["classification"] == "typical"