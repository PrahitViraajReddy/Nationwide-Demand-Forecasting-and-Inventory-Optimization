"""M5 local data checks and loading helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_FILES = ("calendar.csv", "sell_prices.csv")
SALES_CANDIDATES = ("sales_train_validation.csv", "sales_train_evaluation.csv")


def resolve_sales_file(data_dir: Path) -> Path:
    """Return the first available M5 sales file or raise an actionable error."""
    for name in SALES_CANDIDATES:
        candidate = data_dir / name
        if candidate.is_file():
            return candidate
    expected = ", ".join(SALES_CANDIDATES)
    raise FileNotFoundError(f"No M5 sales file found in {data_dir}. Expected one of: {expected}")


def validate_m5_files(data_dir: Path) -> dict[str, Path]:
    """Check required M5 files and return their paths without loading the full data."""
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).is_file()]
    sales_path = None
    try:
        sales_path = resolve_sales_file(data_dir)
    except FileNotFoundError:
        missing.append("sales_train_validation.csv or sales_train_evaluation.csv")
    if missing:
        raise FileNotFoundError(
            f"Missing required M5 file(s) in {data_dir}: {', '.join(missing)}. "
            "Extract the supplied archive into this directory."
        )
    return {"sales": sales_path, "calendar": data_dir / "calendar.csv", "prices": data_dir / "sell_prices.csv"}


def inspect_m5_files(data_dir: Path) -> dict[str, tuple[int, int]]:
    """Read headers and row counts for a quick local data sanity check."""
    paths = validate_m5_files(data_dir)
    summary: dict[str, tuple[int, int]] = {}
    for key, path in paths.items():
        frame = pd.read_csv(path, nrows=5)
        # Count rows in chunks so the large sales file is not loaded into memory.
        row_count = sum(len(chunk) for chunk in pd.read_csv(path, chunksize=100_000))
        summary[key] = (row_count, len(frame.columns))
        print(f"{key:8s} | rows={row_count:,} | columns={len(frame.columns):,} | file={path.name}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and inspect local M5 dataset files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/m5"))
    args = parser.parse_args()
    inspect_m5_files(args.data_dir)


if __name__ == "__main__":
    main()
