import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from market_robustness.simulation.bootstrap import block_bootstrap


@pytest.fixture
def sample_returns():
    return pd.Series([0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.00, 0.02, -0.03, 0.01])

def test__output_length(sample_returns):
    simulated = block_bootstrap(sample_returns, 3, 9)
    assert len(sample_returns) == len(simulated)


def test_reproducibility(sample_returns):
    simulated1 = block_bootstrap(sample_returns,3,9)
    simulated2 = block_bootstrap(sample_returns,3,9)
    assert_series_equal(simulated1,simulated2)

def test_different_seeds(sample_returns):
    simulated1 = block_bootstrap(sample_returns,3,9)
    simulated2 = block_bootstrap(sample_returns,3,10)
    assert not simulated1.equals(simulated2)

def test_output_is_series(sample_returns):
    simulated = block_bootstrap(sample_returns, 3, 9)
    assert isinstance(simulated, pd.Series)

