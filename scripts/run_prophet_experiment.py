"""Compare Prophet with weekly seasonal naive on a bounded M5 subset.

Example:
  python scripts/run_prophet_experiment.py --series-limit 10 --horizon 28
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from demand_inventory.forecasting import load_m5_subset
from demand_inventory.metrics import wape
from demand_inventory.prophet_model import forecast_with_prophet, interval_metrics


def run(data_dir: Path, output_dir: Path, series_limit: int = 10, horizon: int = 28) -> dict:
    if series_limit < 1 or horizon < 1:
        raise ValueError("series_limit and horizon must be at least 1")
    frame = load_m5_subset(data_dir, series_limit=series_limit)
    dates = sorted(frame["date"].unique())
    if len(dates) <= horizon + 30:
        raise ValueError("Not enough history for Prophet training and holdout")
    cutoff = pd.Timestamp(dates[-horizon])
    results = []
    for series_id, group in frame.groupby("id", observed=True, sort=True):
        group = group.sort_values("date")
        train = group[group["date"] < cutoff][["date", "sales"]]
        actual = group[group["date"] >= cutoff][["date", "sales"]]
        if len(actual) != horizon or len(train) < 30:
            continue
        predicted = forecast_with_prophet(train, horizon_days=horizon)
        scored = actual.merge(predicted, on="date", how="inner", validate="one_to_one")
        scored["id"] = series_id
        scored["seasonal_naive"] = np.resize(train["sales"].tail(7).to_numpy(dtype=float), horizon)
        results.append(scored)
    if not results:
        raise ValueError("No series had enough complete training and validation observations")
    scored = pd.concat(results, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = output_dir / "prophet_validation_forecasts.csv"
    scored.to_csv(forecast_path, index=False)
    try:
        prophet_wape = wape(scored["sales"], scored["forecast"])
        naive_wape = wape(scored["sales"], scored["seasonal_naive"])
    except ValueError:
        prophet_wape = naive_wape = None
    interval = interval_metrics(scored["sales"], scored["lower"], scored["upper"])
    metrics = {
        "model": "Prophet",
        "baseline": "weekly seasonal naive",
        "series_count": int(scored["id"].nunique()),
        "horizon_days": int(horizon),
        "validation_start": str(pd.Timestamp(scored["date"].min()).date()),
        "validation_end": str(pd.Timestamp(scored["date"].max()).date()),
        "prophet_wape": prophet_wape,
        "seasonal_naive_wape": naive_wape,
        "interval_coverage": interval["coverage"],
        "mean_interval_width": interval["mean_interval_width"],
        "forecast_file": str(forecast_path),
        "note": "Development subset only; these intervals must be assessed on an untouched test period.",
    }
    (output_dir / "prophet_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/m5"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/prophet"))
    parser.add_argument("--series-limit", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=28)
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output_dir, args.series_limit, args.horizon), indent=2))


if __name__ == "__main__":
    main()
