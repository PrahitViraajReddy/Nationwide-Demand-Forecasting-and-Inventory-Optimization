"""Prepare a compact long-format CSV from the local M5 data for the Dash demo.

Example:
  python scripts/prepare_m5_demo.py --series-limit 20 --output examples/m5_demo_history.csv

The script accepts M5 files at the repository root or under data/raw/m5.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from demand_inventory.data import ID_COLUMNS if False else validate_m5_files
