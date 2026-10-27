import pytest

from demand_inventory.inventory import optimize_replenishment


def test_replenishment_rejects_forecast_that_does_not_cover_window():
    with pytest.raises(ValueError, match="must cover"):
        optimize_replenishment(
            [2, 3],
            on_hand_inventory=1,
            lead_time_days=2,
            review_period_days=1,
        )


def test_replenishment_rejects_negative_inputs():
    with pytest.raises(ValueError, match="non-negative"):
        optimize_replenishment(
            [2, -1, 3],
            on_hand_inventory=1,
            lead_time_days=2,
        )


def test_ortools_replenishment_finds_cost_minimizing_order():
    pytest.importorskip("ortools")
    result = optimize_replenishment(
        [4, 5, 3],
        on_hand_inventory=2,
        lead_time_days=3,
        holding_cost_per_unit=0.1,
        stockout_cost_per_unit=10.0,
    )
    assert result.solver_status == "OPTIMAL"
    assert result.order_quantity == 10
    assert result.projected_shortage == pytest.approx(0.0)
    assert result.modeled_total_cost == pytest.approx(0.0)


def test_replenishment_respects_order_multiple_and_cap():
    pytest.importorskip("ortools")
    result = optimize_replenishment(
        [4, 5, 3],
        on_hand_inventory=0,
        lead_time_days=3,
        holding_cost_per_unit=0.1,
        stockout_cost_per_unit=10.0,
        max_order_quantity=8,
        order_multiple=4,
    )
    assert result.order_quantity in (0, 4, 8)
    assert result.order_quantity <= 8
