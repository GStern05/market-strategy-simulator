import numpy as np
import pandas as pd
import pytest

import market_robustness.data.loader


def test_load_price_data_raises_on_missing_values(mocker):
    fake_df = pd.DataFrame(
        {
            ("Close", "AAPL"): [150.0, np.nan],
            ("High", "AAPL"): [151.0, 152.0],
            ("Low", "AAPL"): [149.0, 150.0],
            ("Open", "AAPL"): [150.5, 151.5],
            ("Volume", "AAPL"): [1000, 1100],
        }
    )
    fake_df.columns = pd.MultiIndex.from_tuples(
        fake_df.columns, names=["Price", "Ticker"]
    )

    mocker.patch(
        "market_robustness.data.loader.yf.download", return_value=fake_df
    )

    with pytest.raises(ValueError):
        market_robustness.data.loader.load_price_data(
            "AAPL", "2026-01-01", "2026-06-01"
        )