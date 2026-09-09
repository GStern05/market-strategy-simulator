import pandas as pd
import pytest

from market_robustness.data.returns import calculate_returns


def test_calculate_returns():
    df = pd.DataFrame({"Close": [100.0, 110.0, 99.0]})
    returns = calculate_returns(df)
    assert returns.iloc[1] == pytest.approx(0.10)
    assert returns.iloc[2] == pytest.approx(-0.10)