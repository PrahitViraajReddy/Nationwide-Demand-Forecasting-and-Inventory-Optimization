import pytest

from demand_inventory.metrics import rmsse, wape


def test_wape_known_value():
    assert wape([10, 20], [8, 22]) == pytest.approx(4 / 30)


def test_wape_rejects_all_zero_actuals():
    with pytest.raises(ValueError, match="all actual values are zero"):
        wape([0, 0], [0, 1])


def test_wape_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        wape([1, 2], [1])


def test_rmsse_known_value():
    # Training scale = mean([1, 1, 1]^2) = 1; forecast RMSE = 1.
    assert rmsse([2, 4], [1, 5], [1, 2, 3, 4], seasonal_period=1) == pytest.approx(1.0)


def test_rmsse_rejects_zero_training_scale():
    with pytest.raises(ValueError, match="training scale is zero"):
        rmsse([1], [2], [5, 5, 5], seasonal_period=1)
