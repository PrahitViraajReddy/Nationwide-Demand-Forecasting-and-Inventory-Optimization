"""Prepare a compact long-format CSV from local M5 data for the Dash demo.

Example:
  python scripts/prepare_m5_demo.py --series-limit 20 --output examples/m5_demo_history.csv

The script accepts M5 files at the repository root or under data/raw/m5.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from demand_inventory.data import validate_m5_files

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def prepare(data_dir: Path, output: Path, series_limit: int = 20) -> Path:
    if series_limit < 1:
        raise ValueError("series_limit must be at least 1")
    paths = validate_m5_files(data_dir)
    sales_path = paths["sales"]
    calendar_path = paths["calendar"]

    metadata = pd.read_csv(sales_path, usecols=["id"])
    count = min(series_limit, len(metadata))
    selected = set(np.unique(np.linspace(0, len(metadata) - 1, count, dtype=int)).tolist())
    day_columns = [column for column in pd.read_csv(sales_path, nrows=0).columns if column.startswith("d_")]
    usecols = ID_COLUMNS + day_columns

    selected_rows = []
    offset = 0
    for chunk in pd.read_csv(sales_path, usecols=usecols, chunksize=128):
        positions = [i for i in range(len(chunk)) if offset + i in selected]
        if positions:
            selected_rows.append(chunk.iloc[positions].copy())
        offset += len(chunk)
        if offset > max(selected):
            break
    if not selected_rows:
        raise ValueError("No M5 sales series were selected")

    wide = pd.concat(selected_rows, ignore_index=True)
    long = wide.melt(
        id_vars=ID_COLUMNS,
        value_vars=day_columns,
        var_name="d",
        value_name="sales",
    )
    calendar = pd.read_csv(calendar_path, usecols=["d", "date"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    long = long.merge(calendar, on="d", how="left", validate="many_to_one")
    if long["date"].isna().any():
        raise ValueError("Some M5 sales day keys did not match calendar.csv")
    long = long[["id", "item_id", "store_id", "date", "sales"]]
    long["sales"] = pd.to_numeric(long["sales"], errors="raise").astype("int32")
    long = long.sort_values(["id", "date"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(output, index=False)
    print(f"Wrote {len(long):,} real M5 observations across {long['id'].nunique():,} series to {output}")
    print("This is observed history only; the dashboard calculates forecasts and validation metrics.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/m5"))
    parser.add_argument("--output", type=Path, default=Path("examples/m5_demo_history.csv"))
    parser.add_argument("--series-limit", type=int, default=20)
    args = parser.parse_args()
    prepare(args.data_dir, args.output, args.series_limit)


if __name__ == "__main__":
    main()
