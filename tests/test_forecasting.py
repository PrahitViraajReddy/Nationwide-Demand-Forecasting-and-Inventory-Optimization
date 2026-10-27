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


def test_load_m5_subset_joins_calendar_and_prices(tmp_path):
    from demand_inventory.forecasting import load_m5_subset

    sales = pd.DataFrame(
        [
            ["item_a_CA_1_validation", "item_a", "dept_a", "cat_a", "CA_1", "CA", *range(10)],
            ["item_b_CA_1_validation", "item_b", "dept_b", "cat_b", "CA_1", "CA", *range(1, 11)],
            ["item_c_TX_1_validation", "item_c", "dept_c", "cat_c", "TX_1", "TX", *range(2, 12)],
        ],
        columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", *[f"d_{i}" for i in range(1, 11)]],
    )
    sales.to_csv(tmp_path / "sales_train_validation.csv", index=False)
    pd.DataFrame(
        {
            "d": [f"d_{i}" for i in range(1, 11)],
            "date": pd.date_range("2025-01-01", periods=10),
            "wm_yr_wk": [11101] * 10,
            "event_name_1": [None] * 10,
            "event_type_1": [None] * 10,
            "event_name_2": [None] * 10,
            "event_type_2": [None] * 10,
            "snap_CA": [0] * 10,
            "snap_TX": [0] * 10,
            "snap_WI": [0] * 10,
        }
    ).to_csv(tmp_path / "calendar.csv", index=False)
    pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1"],
            "item_id": ["item_a", "item_c"],
            "wm_yr_wk": [11101, 11101],
            "sell_price": [2.5, 3.0],
        }
    ).to_csv(tmp_path / "sell_prices.csv", index=False)

    result = load_m5_subset(tmp_path, series_limit=2)
    assert result["id"].nunique() == 2
    assert len(result) == 20
    assert result["date"].min() == pd.Timestamp("2025-01-01")
    assert result.loc[result["item_id"] == "item_a", "sell_price"].eq(2.5).all()
