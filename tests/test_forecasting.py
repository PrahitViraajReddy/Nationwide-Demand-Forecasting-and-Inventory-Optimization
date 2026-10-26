import numpy as np
import pandas as pd
import pytest

from demand_inventory.forecasting import add_history_features


def test_lags_and_rolling_features_use_past_values_only():
    frame = pd.DataFrame({
        "id": ["a"] * 35,
        "date": pd.date_range("2020-01-01", periods=35),
        "sales": np.arange(35, dtype=float),
    })
    result = add_history_features(frame)
    row = result.loc[result["date"] == pd.Timestamp("2020-02-04")].iloc[0]
    assert row["lag_1"] == 33
    assert row["lag_7"] == 27
    assert row["rolling_mean_7"] == pytest.approx(np.mean(np.arange(27, 34)))


def test_lag_features_are_missing_at_start_of_series():
    frame = pd.DataFrame({
        "id": ["a"] * 3,
        "date": pd.date_range("2020-01-01", periods=3),
        "sales": [1.0, 2.0, 3.0],
    })
    result = add_history_features(frame)
    assert result["lag_7"].isna().all()
