"""Cost-based replenishment optimization using OR-Tools.

The objective is an explicit decision model, not a claim of real-world savings.
M5 does not provide on-hand inventory, lead times, or cost parameters; callers
must supply those inputs and document any assumptions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ReplenishmentDecision:
    order_quantity: int
    expected_demand: float
    on_hand_inventory: float
    projected_ending_inventory: float
    projected_shortage: float
    modeled_holding_cost: float
    modeled_stockout_cost: float
    modeled_total_cost: float
    lead_time_days: int
    review_period_days: int
    order_multiple: int
    solver_status: str

    def to_dict(self) -> dict:
        return asdict(self)


def optimize_replenishment(
    demand_forecast: Sequence[float],
    *,
    on_hand_inventory: float,
    lead_time_days: int,
    review_period_days: int = 0,
    holding_cost_per_unit: float = 0.01,
    stockout_cost_per_unit: float = 1.0,
    max_order_quantity: int | None = None,
    order_multiple: int = 1,
) -> ReplenishmentDecision:
    """Minimize modeled holding + shortage cost for demand over the coverage window.

    Forecast values should cover lead time plus the review period. The objective
    uses deterministic forecast demand and linear per-unit costs; it does not
    represent purchase cost, supplier capacity, or stochastic service levels.
    """
    demand = np.asarray(demand_forecast, dtype=float).reshape(-1)
    if demand.size == 0 or not np.isfinite(demand).all():
        raise ValueError("demand_forecast must contain finite values")
    if (demand < 0).any():
        raise ValueError("demand_forecast must be non-negative")
    if on_hand_inventory < 0 or not np.isfinite(on_hand_inventory):
        raise ValueError("on_hand_inventory must be finite and non-negative")
    if lead_time_days < 0 or review_period_days < 0:
        raise ValueError("lead_time_days and review_period_days must be non-negative")
    if lead_time_days + review_period_days > len(demand):
        raise ValueError("demand_forecast must cover lead_time_days + review_period_days")
    if holding_cost_per_unit < 0 or stockout_cost_per_unit < 0:
        raise ValueError("cost parameters must be non-negative")
    if not np.isfinite(holding_cost_per_unit) or not np.isfinite(stockout_cost_per_unit):
        raise ValueError("cost parameters must be finite")
    if order_multiple < 1:
        raise ValueError("order_multiple must be at least 1")
    if max_order_quantity is not None and max_order_quantity < 0:
        raise ValueError("max_order_quantity must be non-negative")

    coverage_days = lead_time_days + review_period_days
    expected_demand = float(demand[:coverage_days].sum()) if coverage_days else 0.0
    max_order = int(max_order_quantity) if max_order_quantity is not None else int(np.ceil(expected_demand))
    max_packs = floor(max_order / order_multiple)

    try:
        from ortools.linear_solver import pywraplp
    except ImportError as exc:
        raise ImportError(
            "OR-Tools is required for replenishment optimization. Install it with: pip install ortools"
        ) from exc

    solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("No supported OR-Tools MIP solver is available (SCIP/CBC).")

    packs = solver.IntVar(0, max_packs, "order_packs")
    order_quantity = order_multiple * packs
    ending_inventory = solver.NumVar(0.0, solver.infinity(), "ending_inventory")
    shortage = solver.NumVar(0.0, solver.infinity(), "shortage")
    solver.Add(ending_inventory >= on_hand_inventory + order_quantity - expected_demand)
    solver.Add(shortage >= expected_demand - on_hand_inventory - order_quantity)
    objective = solver.Objective()
    objective.SetCoefficient(ending_inventory, float(holding_cost_per_unit))
    objective.SetCoefficient(shortage, float(stockout_cost_per_unit))
    objective.SetMinimization()

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        status_name = {
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
            pywraplp.Solver.ABNORMAL: "ABNORMAL",
            pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
        }.get(status, f"STATUS_{status}")
        raise RuntimeError(f"Replenishment solver did not find an optimal solution: {status_name}")

    order = int(round(order_quantity.solution_value()))
    ending = max(0.0, on_hand_inventory + order - expected_demand)
    unmet = max(0.0, expected_demand - on_hand_inventory - order)
    holding_cost = ending * holding_cost_per_unit
    shortage_cost = unmet * stockout_cost_per_unit
    return ReplenishmentDecision(
        order_quantity=order,
        expected_demand=expected_demand,
        on_hand_inventory=float(on_hand_inventory),
        projected_ending_inventory=ending,
        projected_shortage=unmet,
        modeled_holding_cost=holding_cost,
        modeled_stockout_cost=shortage_cost,
        modeled_total_cost=holding_cost + shortage_cost,
        lead_time_days=int(lead_time_days),
        review_period_days=int(review_period_days),
        order_multiple=int(order_multiple),
        solver_status="OPTIMAL",
    )
