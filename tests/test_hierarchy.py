import pandas as pd
import pytest

from demand_inventory.hierarchy import aggregate_bottom_up, coherence_error


def _forecast_frame():
    return pd.DataFrame(
        {
            "id": ["a_CA1", "b_CA1", "a_CA2", "a_CA1", "b_CA1", "a_CA2"],
            "store_id": ["CA_1", "CA_1", "CA_2", "CA_1", "CA_1", "CA_2"],
            "state_id": ["CA", "CA", "CA", "CA", "CA", "CA"],
            "date": pd.to_datetime(
                ["2025-01-01"] * 3 + ["2025-01-02"] * 3
            ),
            "forecast": [1, 2, 4, 2, 3, 5],
        }
    )


def test_bottom_up_aggregation_has_expected_store_state_and_total_values():
    result = aggregate_bottom_up(_forecast_frame())
    day = result[result["date"] == pd.Timestamp("2025-01-01")]
    store = day[(day["node_level"] == "store") & (day["node_id"] == "CA_1")]["forecast"]
    state = day[day["node_level"] == "state"]["forecast"]
    total = day[day["node_level"] == "total"]["forecast"]
    assert store.iloc[0] == 3
    assert state.iloc[0] == 7
    assert total.iloc[0] == 7


def test_bottom_up_aggregation_is_coherent():
    assert coherence_error(_forecast_frame()) == pytest.approx(0.0)


def test_hierarchy_rejects_duplicate_series_dates():
    frame = _forecast_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="must be unique"):
        aggregate_bottom_up(frame)
