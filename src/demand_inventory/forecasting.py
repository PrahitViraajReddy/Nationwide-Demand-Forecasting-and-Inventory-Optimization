"""Leakage-safe M5 subset loader, seasonal-naive baseline, and LightGBM forecast.

This first experiment intentionally uses a reproducible, evenly spaced subset of
series. It is a development workflow, not a claim of full-competition accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import validate_m5_files
from .metrics import rmsse, wape

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
LAG_DAYS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 28)
EVENT_COLUMNS = ("event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI")


def load_m5_subset(data_dir: Path, series_limit: int = 100) -> pd.DataFrame:
    """Load a deterministic, evenly spaced subset from the wide M5 sales file."""
    if series_limit < 1:
        raise ValueError("series_limit must be at least 1")
    paths = validate_m5_files(data_dir)
    sales_path = paths["sales"]

    metadata = pd.read_csv(sales_path, usecols=ID_COLUMNS)
    take = min(series_limit, len(metadata))
    selected = np.unique(np.linspace(0, len(metadata) - 1, take, dtype=int))
    selected_set = set(selected.tolist())
    day_columns = [c for c in pd.read_csv(sales_path, nrows=0).columns if c.startswith("d_")]
    keep = set(ID_COLUMNS + day_columns)

    selected_frames = []
    offset = 0
    for chunk in pd.read_csv(sales_path, chunksize=256, usecols=lambda c: c in keep):
        local_positions = [i for i in range(len(chunk)) if offset + i in selected_set]
        if local_positions:
            selected_frames.append(chunk.iloc[local_positions].copy())
        offset += len(chunk)
        if offset > selected[-1]:
            break
    if not selected_frames:
        raise ValueError(f"No sales series selected from {sales_path}")

    wide = pd.concat(selected_frames, ignore_index=True)
    long = wide.melt(id_vars=ID_COLUMNS, var_name="d", value_name="sales")
    long["sales"] = pd.to_numeric(long["sales"], errors="raise").astype("float32")

    calendar = pd.read_csv(paths["calendar"])
    calendar_columns = ["d", "date", "wm_yr_wk"] + [c for c in EVENT_COLUMNS if c in calendar.columns]
    calendar = calendar[calendar_columns].copy()
    calendar["date"] = pd.to_datetime(calendar["date"])
    long = long.merge(calendar, on="d", how="left", validate="many_to_one")
    if long["date"].isna().any():
        raise ValueError("Some sales day keys did not match calendar.csv")

    prices = pd.read_csv(paths["prices"], usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"])
    long = long.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    long = long.sort_values(["id", "date"]).reset_index(drop=True)
    for col in EVENT_COLUMNS:
        if col in long.columns:
            if long[col].dtype == object:
                long[col] = long[col].fillna("None").astype("category")
            else:
                long[col] = long[col].fillna(0).astype("int8")
    long["sell_price"] = long["sell_price"].astype("float32")
    long["series_id"] = long["id"].astype("category")
    return long


def add_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create lag and shifted rolling features without reading future sales."""
    frame = frame.sort_values(["id", "date"]).copy()
    grouped = frame.groupby("id", observed=True)["sales"]
    for lag in LAG_DAYS:
        frame[f"lag_{lag}"] = grouped.shift(lag)
    for window in ROLLING_WINDOWS:
        frame[f"rolling_mean_{window}"] = frame.groupby("id", observed=True)["sales"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean()
        )
        frame[f"rolling_std_{window}"] = frame.groupby("id", observed=True)["sales"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).std()
        )
    frame["day_of_week"] = frame["date"].dt.dayofweek.astype("int8")
    frame["day_of_month"] = frame["date"].dt.day.astype("int8")
    frame["month"] = frame["date"].dt.month.astype("int8")
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype("int16")
    frame["year"] = frame["date"].dt.year.astype("int16")
    return frame


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    base = [
        "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "sell_price", "day_of_week", "day_of_month", "month", "week_of_year", "year",
    ]
    base += [f"lag_{lag}" for lag in LAG_DAYS]
    base += [f"rolling_mean_{window}" for window in ROLLING_WINDOWS]
    base += [f"rolling_std_{window}" for window in ROLLING_WINDOWS]
    base += [c for c in EVENT_COLUMNS if c in frame.columns]
    return base


def _encode_categories(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align category codes across training and prediction rows."""
    for column in train_x.columns:
        if train_x[column].dtype.name == "category" or train_x[column].dtype == object:
            values = pd.concat([train_x[column].astype(str), test_x[column].astype(str)], ignore_index=True)
            categories = pd.Index(values.unique())
            train_x[column] = pd.Categorical(train_x[column].astype(str), categories=categories)
            test_x[column] = pd.Categorical(test_x[column].astype(str), categories=categories)
    return train_x, test_x


def _recursive_features(
    histories: dict[str, list[float]],
    static: pd.DataFrame,
    future_calendar: pd.DataFrame,
    price_lookup: dict[tuple[str, str, int], float],
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build one-day-ahead feature rows using observed history plus prior predictions."""
    calendar_row = future_calendar.loc[future_calendar["date"] == target_date]
    if calendar_row.empty:
        raise ValueError(f"No calendar row for forecast date {target_date.date()}")
    calendar_row = calendar_row.iloc[0]
    rows = []
    for _, meta in static.iterrows():
        series_id = meta["id"]
        history = histories[series_id]
        row = {column: meta[column] for column in ID_COLUMNS}
        row["id"] = series_id
        row["date"] = target_date
        row["d"] = calendar_row["d"]
        row["wm_yr_wk"] = calendar_row["wm_yr_wk"]
        row["sales"] = np.nan
        for col in EVENT_COLUMNS:
            if col in future_calendar.columns:
                row[col] = calendar_row[col]
        price_key = (meta["store_id"], meta["item_id"], int(calendar_row["wm_yr_wk"]))
        row["sell_price"] = price_lookup.get(price_key, np.nan)
        for lag in LAG_DAYS:
            row[f"lag_{lag}"] = history[-lag] if len(history) >= lag else np.nan
        for window in ROLLING_WINDOWS:
            values = history[-window:]
            row[f"rolling_mean_{window}"] = float(np.mean(values)) if len(values) == window else np.nan
            row[f"rolling_std_{window}"] = float(np.std(values, ddof=1)) if len(values) == window else np.nan
        row["day_of_week"] = target_date.dayofweek
        row["day_of_month"] = target_date.day
        row["month"] = target_date.month
        row["week_of_year"] = int(target_date.isocalendar().week)
        row["year"] = target_date.year
        rows.append(row)
    return pd.DataFrame(rows)


def run_lightgbm_experiment(
    data_dir: Path,
    output_dir: Path,
    series_limit: int = 100,
    horizon: int = 28,
    random_seed: int = 42,
) -> dict:
    """Train a global LightGBM model and compare it with weekly seasonal naive."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    frame = load_m5_subset(data_dir, series_limit=series_limit)
    frame = add_history_features(frame)
    dates = np.sort(frame["date"].unique())
    if len(dates) <= max(LAG_DAYS) + horizon:
        raise ValueError("Not enough history for the requested lags and forecast horizon")
    cutoff = pd.Timestamp(dates[-horizon])
    train = frame[(frame["date"] < cutoff)].dropna(subset=[f"lag_{max(LAG_DAYS)}", "rolling_mean_28"]).copy()
    actual = frame[frame["date"] >= cutoff].copy()
    if train.empty or actual.empty:
        raise ValueError("Training or validation set is empty after feature preparation")

    feature_columns = _feature_columns(frame)
    train_x = train[feature_columns].copy()
    train_y = train["sales"].astype(float)
    actual_x = actual[feature_columns].copy()
    train_x, actual_x = _encode_categories(train_x, actual_x)

    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train_x, train_y, categorical_feature="auto")

    # Recursive forecasts: later-day lags use prior predictions, never validation actuals.
    static = frame[ID_COLUMNS].drop_duplicates("id").sort_values("id").reset_index(drop=True)
    histories = {
        series_id: group.loc[group["date"] < cutoff, "sales"].astype(float).tolist()
        for series_id, group in frame.groupby("id", observed=True, sort=False)
    }
    paths = validate_m5_files(data_dir)
    calendar = pd.read_csv(paths["calendar"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    future_calendar = calendar[calendar["date"] >= cutoff].copy()
    future_prices = pd.read_csv(paths["prices"], usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"])
    selected_keys = set(zip(static["store_id"], static["item_id"]))
    future_prices = future_prices[
        future_prices.apply(lambda row: (row["store_id"], row["item_id"]) in selected_keys, axis=1)
    ]
    price_lookup = {
        (row.store_id, row.item_id, int(row.wm_yr_wk)): float(row.sell_price)
        for row in future_prices.itertuples(index=False)
    }
    predicted_rows = []
    for target_date in sorted(actual["date"].unique()):
        date = pd.Timestamp(target_date)
        batch = _recursive_features(histories, static, future_calendar, price_lookup, date)
        x = batch[feature_columns].copy()
        # Reapply the exact categorical levels seen at training.
        for column in train_x.columns:
            if isinstance(train_x[column].dtype, pd.CategoricalDtype):
                x[column] = pd.Categorical(x[column].astype(str), categories=train_x[column].cat.categories)
        predictions = np.maximum(0.0, model.predict(x))
        batch["forecast"] = predictions
        for series_id, prediction in zip(batch["id"], predictions):
            histories[series_id].append(float(prediction))
        predicted_rows.append(batch[["id", "item_id", "store_id", "state_id", "date", "forecast"]])
    forecast = pd.concat(predicted_rows, ignore_index=True)
    actual_values = actual[["id", "date", "sales"]].rename(columns={"sales": "actual"})
    scored = forecast.merge(actual_values, on=["id", "date"], how="left", validate="one_to_one")

    # Weekly seasonal-naive forecast repeats the most recent matching weekday.
    observed_history = frame[frame["date"] < cutoff][["id", "date", "sales"]]
    last_28 = observed_history.sort_values("date").groupby("id", observed=True).tail(28)
    last_values = {sid: group["sales"].tolist() for sid, group in last_28.groupby("id", observed=True)}
    naive_values = []
    for _, row in scored.iterrows():
        values = last_values[row["id"]]
        step = (row["date"] - cutoff).days
        naive_values.append(float(values[-7 + (step % 7)]) if len(values) >= 7 else float(values[-1]))
    scored["seasonal_naive"] = naive_values

    train_series = {
        sid: group.sort_values("date")["sales"].to_numpy(dtype=float)
        for sid, group in observed_history.groupby("id", observed=True)
    }
    model_rmsse, naive_rmsse = [], []
    for sid, group in scored.groupby("id", observed=True):
        history = train_series[sid]
        try:
            model_rmsse.append(rmsse(group["actual"], group["forecast"], history, seasonal_period=7))
            naive_rmsse.append(rmsse(group["actual"], group["seasonal_naive"], history, seasonal_period=7))
        except ValueError:
            continue

    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "lightgbm_validation_forecasts.csv"
    scored.to_csv(scored_path, index=False)
    metrics = {
        "model": "LightGBM recursive one-day-ahead",
        "baseline": "weekly seasonal naive",
        "series_count": int(scored["id"].nunique()),
        "horizon_days": int(horizon),
        "validation_start": str(pd.Timestamp(scored["date"].min()).date()),
        "validation_end": str(pd.Timestamp(scored["date"].max()).date()),
        "lightgbm_wape": wape(scored["actual"], scored["forecast"]),
        "seasonal_naive_wape": wape(scored["actual"], scored["seasonal_naive"]),
        "lightgbm_mean_series_rmsse": float(np.mean(model_rmsse)) if model_rmsse else None,
        "seasonal_naive_mean_series_rmsse": float(np.mean(naive_rmsse)) if naive_rmsse else None,
        "forecast_file": str(scored_path),
        "note": "Development subset only; metrics are not full M5 leaderboard results.",
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first M5 LightGBM forecasting experiment.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/m5"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--series-limit", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_lightgbm_experiment(args.data_dir, args.output_dir, args.series_limit, args.horizon, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
