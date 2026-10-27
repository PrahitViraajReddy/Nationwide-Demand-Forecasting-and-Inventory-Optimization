"""M5 data discovery, integrity checks, and lightweight inspection helpers."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_FILES = ("calendar.csv", "sell_prices.csv")
SALES_CANDIDATES = ("sales_train_validation.csv", "sales_train_evaluation.csv")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _has_m5_layout(directory: Path) -> bool:
    return (
        all((directory / name).is_file() for name in REQUIRED_FILES)
        and any((directory / name).is_file() for name in SALES_CANDIDATES)
    )


def resolve_data_dir(data_dir: Path) -> Path:
    """Resolve an M5 directory, supporting both repo-root LFS files and local data/raw/m5."""
    requested = Path(data_dir).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(REPO_ROOT / requested)
    candidates.extend([REPO_ROOT / "data" / "raw" / "m5", REPO_ROOT])
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen and _has_m5_layout(candidate):
            return candidate
        seen.add(candidate)
    expected = "calendar.csv, sell_prices.csv, and sales_train_validation.csv or sales_train_evaluation.csv"
    raise FileNotFoundError(
        f"Could not find an M5 dataset directory. Checked: {', '.join(map(str, seen))}. "
        f"Expected {expected}. The repository supports files at its root or under data/raw/m5."
    )


def _reject_lfs_pointer(path: Path) -> None:
    """Fail clearly when Git checked out an LFS pointer instead of the actual CSV."""
    try:
        with path.open("rb") as stream:
            prefix = stream.read(128)
    except OSError as exc:
        raise OSError(f"Cannot read M5 data file: {path}") from exc
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the dataset contents. From the repository root run "
            "'git lfs install' and 'git lfs pull', then retry. If running in deployment, ensure the "
            "build environment downloads Git LFS objects or provide the data through external storage."
        )


def resolve_sales_file(data_dir: Path) -> Path:
    """Return the preferred available M5 sales file or raise an actionable error."""
    for name in SALES_CANDIDATES:
        candidate = data_dir / name
        if candidate.is_file():
            _reject_lfs_pointer(candidate)
            return candidate
    expected = ", ".join(SALES_CANDIDATES)
    raise FileNotFoundError(f"No M5 sales file found in {data_dir}. Expected one of: {expected}")


def validate_m5_files(data_dir: Path) -> dict[str, Path]:
    """Resolve and validate the required M5 files without loading the full tables."""
    resolved_dir = resolve_data_dir(data_dir)
    missing = [name for name in REQUIRED_FILES if not (resolved_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required M5 file(s) in {resolved_dir}: {', '.join(missing)}")
    sales_path = resolve_sales_file(resolved_dir)
    calendar_path = resolved_dir / "calendar.csv"
    prices_path = resolved_dir / "sell_prices.csv"
    for path in (calendar_path, prices_path):
        _reject_lfs_pointer(path)
    return {"sales": sales_path, "calendar": calendar_path, "prices": prices_path}


def inspect_m5_files(data_dir: Path) -> dict[str, tuple[int, int]]:
    """Read headers and row counts in chunks so wide sales data is not loaded at once."""
    paths = validate_m5_files(data_dir)
    summary: dict[str, tuple[int, int]] = {}
    for key, path in paths.items():
        frame = pd.read_csv(path, nrows=5)
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
