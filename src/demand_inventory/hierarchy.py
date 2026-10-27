"""Store/state hierarchy aggregation and optional Darts MinT reconciliation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "store_id", "state_id", "date", "forecast"}
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError(f"forecasts is missing required columns: {', '.join(sorted(missing))}")
    frame = forecasts[list(required)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["forecast"] = pd.to_numeric(frame["forecast"], errors="coerce")
    if frame[["date", "forecast"]].isna().any().any():
        raise ValueError("forecast dates and values must be valid and non-null")
    if not np.isfinite(frame["forecast"]).all():
        raise ValueError("forecast values must be finite")
    if frame.duplicated(["id", "date"]).any():
        raise ValueError("each id/date pair must be unique")
    if frame[["id", "store_id", "state_id"]].isna().any().any():
        raise ValueError("id, store_id, and state_id cannot be null")
    return frame


def aggregate_bottom_up(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bottom-level item-store forecasts to store, state, and total.

    Returns a long table with node_id, node_level, date, and forecast. The total
    is the sum of the selected bottom-level series; callers must ensure the
    selected series represent the population they intend to report.
    """
    frame = _validate_forecasts(forecasts)
    bottom = frame.assign(node_id=frame["id"].astype(str), node_level="item_store")
    store = (
        frame.groupby(["store_id", "date"], as_index=False, observed=True)["forecast"].sum()
        .rename(columns={"store_id": "node_id"})
        .assign(node_level="store")
    )
    state = (
        frame.groupby(["state_id", "date"], as_index=False, observed=True)["forecast"].sum()
        .rename(columns={"state_id": "node_id"})
        .assign(node_level="state")
    )
    total = (
        frame.groupby("date", as_index=False, observed=True)["forecast"].sum()
        .assign(node_id="total", node_level="total")
    )
    return pd.concat(
        [
            bottom[["node_id", "node_level", "date", "forecast"]],
            store[["node_id", "node_level", "date", "forecast"]],
            state[["node_id", "node_level", "date", "forecast"]],
            total[["node_id", "node_level", "date", "forecast"]],
        ],
        ignore_index=True,
    ).sort_values(["date", "node_level", "node_id"]).reset_index(drop=True)


def coherence_error(forecasts: pd.DataFrame) -> float:
    """Return maximum absolute store/state/total aggregation discrepancy."""
    frame = _validate_forecasts(forecasts)
    aggregated = aggregate_bottom_up(frame)
    max_error = 0.0
    for date, daily in frame.groupby("date", observed=True):
        nodes = aggregated.loc[aggregated["date"] == date]
        total = float(daily["forecast"].sum())
        reported_total = nodes.loc[nodes["node_level"] == "total", "forecast"]
        if reported_total.empty:
            raise ValueError(f"Missing total aggregate for {date.date()}")
        max_error = max(max_error, abs(float(reported_total.iloc[0]) - total))
        # Check each store and state against the bottom-level rows it contains.
        for column, level in (("store_id", "store"), ("state_id", "state")):
            expected = daily.groupby(column, observed=True)["forecast"].sum()
            actual = nodes.loc[nodes["node_level"] == level].set_index("node_id")["forecast"]
            for key, value in expected.items():
                max_error = max(max_error, abs(float(actual.loc[key]) - float(value)))
    return float(max_error)


def reconcile_with_darts(forecasts: pd.DataFrame, *, method: str = "ols") -> pd.DataFrame:
    """Reconcile bottom-level item-store forecasts with Darts MinT.

    Darts requires every hierarchy node to be a component of the same TimeSeries.
    The hierarchy used here is item-store -> store -> state -> total. The returned
    long table includes reconciled forecasts for all included nodes.
    """
    frame = _validate_forecasts(forecasts)
    try:
        from darts import TimeSeries
        from darts.dataprocessing.transformers.reconciliation import MinTReconciliator
    except ImportError as exc:
        raise ImportError(
            "Darts is required for MinT reconciliation. Install the project forecasting dependencies."
        ) from exc

    frame["id"] = frame["id"].astype(str)
    frame["store_id"] = frame["store_id"].astype(str)
    frame["state_id"] = frame["state_id"].astype(str)
    frame = frame.sort_values(["date", "id"])
    if frame.groupby("date", observed=True)["id"].nunique().nunique() != 1:
        raise ValueError("Every forecast date must contain the same set of item-store series")

    series_nodes = {series_id: f"item_store::{series_id}" for series_id in frame["id"].unique()}
    store_nodes = {store_id: f"store::{store_id}" for store_id in frame["store_id"].unique()}
    state_nodes = {state_id: f"state::{state_id}" for state_id in frame["state_id"].unique()}
    total_node = "total"
    hierarchy: dict[str, list[str]] = {}
    for row in frame[["id", "store_id", "state_id"]].drop_duplicates().itertuples(index=False):
        hierarchy[series_nodes[row.id]] = [store_nodes[row.store_id]]
    for store_id, state_id in frame[["store_id", "state_id"]].drop_duplicates().itertuples(index=False):
        hierarchy[store_nodes[store_id]] = [state_nodes[state_id]]
    for state_id in state_nodes:
        hierarchy[state_nodes[state_id]] = [total_node]

    bottom = frame.assign(node_id=frame["id"].map(series_nodes))
    store = (
        frame.assign(node_id=frame["store_id"].map(store_nodes))
        .groupby(["date", "node_id"], as_index=False, observed=True)["forecast"].sum()
    )
    state = (
        frame.assign(node_id=frame["state_id"].map(state_nodes))
        .groupby(["date", "node_id"], as_index=False, observed=True)["forecast"].sum()
    )
    total = frame.groupby("date", as_index=False, observed=True)["forecast"].sum().assign(node_id=total_node)
    all_nodes = pd.concat(
        [
            bottom[["date", "node_id", "forecast"]],
            store[["date", "node_id", "forecast"]],
            state[["date", "node_id", "forecast"]],
            total[["date", "node_id", "forecast"]],
        ],
        ignore_index=True,
    )
    wide = all_nodes.pivot(index="date", columns="node_id", values="forecast").sort_index()
    node_order = list(hierarchy.keys())
    node_order.append(total_node)
    node_order = list(dict.fromkeys(node_order))
    wide = wide[node_order]
    if wide.isna().any().any():
        raise ValueError("Forecast hierarchy has missing node/date values")
    series = TimeSeries.from_dataframe(
        wide.reset_index(),
        time_col="date",
        value_cols=node_order,
        freq="D",
        hierarchy=hierarchy,
    )
    reconciled = MinTReconciliator(method=method).fit_transform(series)
    values = reconciled.pd_dataframe().rename_axis("date").reset_index()
    result = values.melt(id_vars="date", var_name="node_id", value_name="forecast")
    result["node_level"] = result["node_id"].map(
        lambda node: "item_store" if node.startswith("item_store::")
        else "store" if node.startswith("store::")
        else "state" if node.startswith("state::")
        else "total"
    )
    result["node_id"] = result["node_id"].str.replace(r"^(item_store|store|state)::", "", regex=True)
    # Do not clip individual reconciled nodes here: that would break additivity.
    # If non-negativity is required, clamp bottom-level forecasts and re-aggregate.
    return result.sort_values(["date", "node_level", "node_id"]).reset_index(drop=True)
