"""Run a bounded TFT experiment on a reproducible M5 subset.

Example:
  python scripts/run_tft_experiment.py --series-limit 50 --horizon 28 --max-epochs 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from demand_inventory.forecasting import load_m5_subset
from demand_inventory.tft_model import train_tft_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/m5"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tft"))
    parser.add_argument("--series-limit", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--encoder-length", type=int, default=56)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = load_m5_subset(args.data_dir, series_limit=args.series_limit)
    result = train_tft_experiment(
        frame,
        args.output_dir,
        horizon=args.horizon,
        max_encoder_length=args.encoder_length,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
