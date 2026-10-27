import pandas as pd
import pytest

from demand_inventory.tft_model import prepare_tft_frame, train_tft_experiment


def test_prepare_tft_frame_adds_time_and_calendar_features():
    frame = pd.DataFrame(
        {
            "id": ["a"] * 20,
            "date": pd.date_range("2025-01-01", periods=20),
            "sales": range(20),
            "store_id": ["CA_1"] * 20,
            "sell_price": [1.0] * 20,
            "event_name_1": [None] * 20,
        }
    )
    result, known_reals, known_categoricals, static_categoricals = prepare_tft_frame(frame)
    assert result["time_idx"].tolist() == list(range(20))
    assert {"time_idx", "day_of_week", "month", "sell_price"}.issubset(known_reals)
    assert known_categoricals == ["event_name_1"]
    assert static_categoricals == ["store_id"]
    assert result["event_name_1"].eq("None").all()


def test_prepare_tft_frame_rejects_duplicate_series_dates():
    frame = pd.DataFrame({"id": ["a", "a"], "date": ["2025-01-01"] * 2, "sales": [1, 2]})
    with pytest.raises(ValueError, match="one row per id/date"):
        prepare_tft_frame(frame)


def test_tft_experiment_validates_horizon_before_optional_dependencies(tmp_path):
    frame = pd.DataFrame({"id": ["a"], "date": ["2025-01-01"], "sales": [1]})
    with pytest.raises(ValueError, match="horizon"):
        train_tft_experiment(frame, tmp_path, horizon=0)
