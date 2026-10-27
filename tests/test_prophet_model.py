import numpy as np
import pandas as pd
import pytest

from demand_inventory.prophet_model import forecast_with_prophet, interval_metrics


def test_interval_metrics_known_coverage_and_width():
    result = interval_metrics([1, 3, 5], [0, 2, 4], [2, 4, 6])
    assert result["coverage"] == pytest.approx(1.0)
    assert result["mean_interval_width"] == pytest.approx(2.0)


def test_interval_metrics_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="lower bounds"):
        interval_metrics([2], [3], [1])


def test_prophet_input_validation_happens_before_optional_import():
    history = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=5), "sales": [1, 2, 1, 3, 2]})
    with pytest.raises(ValueError, match="at least 14"):
        forecast_with_prophet(history)


def test_prophet_rejects_duplicate_dates():
    history = pd.DataFrame({"date": ["2025-01-01"] * 14, "sales": np.ones(14)})
    with pytest.raises(ValueError, match="at most one"):
        forecast_with_prophet(history)
