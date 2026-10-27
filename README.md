# Nationwide Demand Forecasting & Inventory Optimization

A retail decision-support platform for forecasting product demand across the M5 Walmart hierarchy and converting forecasts into constrained replenishment recommendations.

> **Project status:** Foundation in progress. Model accuracy, forecast intervals, reconciliation quality, and inventory savings will be reported only after reproducible experiments; no performance figures are fabricated.

## Problem

Retail teams need to anticipate demand at item–store level while understanding roll-ups by department, category, store, and state. Forecasts should communicate uncertainty and support practical order decisions—not stop at a predicted sales number.

## Planned capabilities

- **Data foundation:** M5 sales, calendar, and sell-price data with schema and integrity checks.
- **Baselines:** Naive/seasonal-naive forecasts and Prophet.
- **Machine learning:** LightGBM with leakage-safe lag, rolling, calendar, holiday, and price features.
- **Deep learning:** Temporal Fusion Transformer using PyTorch Forecasting, evaluated against simpler models.
- **Hierarchical forecasts:** Darts-based reconciliation experiments; aggregate consistency checks.
- **Uncertainty:** Prediction intervals and empirical coverage evaluation.
- **Inventory decisions:** OR-Tools optimization with explicit lead time, service-level, order, and holding/shortage-cost assumptions.
- **Decision interface:** Plotly Dash for forecast exploration, model evaluation, uncertainty, and order recommendations.
- **Quality:** Time-based validation, reproducible configuration, unit tests, and documented limitations.

## Technology stack

Python · pandas · NumPy · scikit-learn · Prophet · LightGBM · Darts · PyTorch Forecasting · OR-Tools · Plotly Dash · pytest

## M5 data

The repository currently tracks the five supplied CSVs through **Git LFS**: `calendar.csv`, `sales_train_validation.csv`, `sales_train_evaluation.csv`, `sell_prices.csv`, and `sample_submission.csv`. The code supports either this repository-root layout or a local `data/raw/m5/` directory.

Install Git LFS before cloning or pulling the data:

```bash
git lfs install
git lfs pull
```

If a CSV begins with `version https://git-lfs.github.com/spec/v1`, Git has checked out the pointer rather than the data. Run the commands above before running the pipeline. For deployment, ensure the build environment downloads the LFS objects or configure external data storage; do not assume that a pointer file is a readable CSV.

The dataset is large. Keep processed data, generated features, checkpoints, and model artifacts out of ordinary Git commits.

## Quick start

Python 3.10–3.11 is recommended for compatibility across the forecasting ecosystem.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Check that the local M5 files are present and readable:

```bash
python -m demand_inventory.data --data-dir data/raw/m5
```

Run tests:

```bash
pytest
```

Launch the interactive baseline dashboard with optional M5 sample loading and OR-Tools replenishment:

```bash
python app.py
``

## Repository layout

```text
app.py
configs/
  default.yaml
data/
  raw/m5/                 # optional local copy; excluded from Git; root CSVs are Git LFS-backed
docs/
  architecture.md
  data-dictionary.md
src/demand_inventory/
  __init__.py
  data.py
  metrics.py
tests/
```

## Evaluation protocol

Use chronological holdouts, never random row splits. Fit feature transformations and models using training history only. Compare against naive and seasonal-naive baselines. Report WAPE and RMSSE with the exact evaluation window and aggregation level. For probabilistic forecasts, report interval coverage and width. Inventory evaluation should include fill rate/stockout proxy, average inventory, and total modeled cost under stated assumptions.

## Important modeling decisions

- M5 sales are intermittent and many item–store series contain zeros; metrics and model choices must account for this.
- Price and calendar features must be joined at the correct item/store/date grain.
- Aggregate consistency is checked across the hierarchy; reconciliation is not assumed to improve accuracy automatically.
- M5 does not provide every operational input needed for real replenishment (for example, true lead times and holding costs). Such inputs must be configurable and clearly labeled as assumptions.
- The full dataset and Temporal Fusion Transformer can be computationally expensive. Begin with a reproducible subset for development, then scale evaluation deliberately.

## Deliverables

- Reproducible forecasting and inventory workflow
- Interactive Plotly Dash application
- Architecture and data documentation
- Automated tests and recorded evaluation results
- Demo recording and 10-minute presentation
- Individual reflection documenting ownership and lessons learned

## License and attribution

The M5 Forecasting dataset is associated with the Walmart forecasting competition. Review the dataset's original terms and attribution requirements before redistribution. The CSVs are stored as Git LFS objects in this repository; the LFS objects must be downloaded before the code can read them.


## First forecasting experiment

After extracting the supplied archive into `data/raw/m5/`, install dependencies and run a reproducible LightGBM development experiment:

```bash
pip install -e ".[dev]"
python -m demand_inventory.forecasting --data-dir data/raw/m5 --series-limit 100 --horizon 28
pytest
```

The loader selects evenly spaced item-store series, joins the M5 calendar and weekly prices, and reads the wide sales table in chunks. It supports CSVs stored at the repository root through Git LFS or under `data/raw/m5/`. The experiment compares a recursive LightGBM forecast with a weekly seasonal-naive baseline. Validation predictions are recursive: after the cutoff, model predictions—not actual validation sales—feed future lag features.

Additional modules provide a Prophet benchmark with interval-coverage metrics (`demand_inventory.prophet_model`), a bounded TFT training experiment (`demand_inventory.tft_model`), bottom-up store/state/total aggregation plus an optional Darts MinT reconciler (`demand_inventory.hierarchy`), and cost-based OR-Tools replenishment (`demand_inventory.inventory`). The Dash interface currently uses the seasonal-naive forecast and OR-Tools replenishment; Prophet, LightGBM, TFT, and Darts workflows are separate experiments and are not yet selectable in the dashboard.

Outputs are written to `artifacts/` (ignored by Git): `metrics.json` and `lightgbm_validation_forecasts.csv`. Results from this limited subset are development metrics, not full M5 benchmark claims.

## Render deployment

For the current interactive Dash demo, set the Render service root to the repository root and use:

- **Build command:** `pip install -r requirements-render.txt`
- **Start command:** `python app.py`

The deployment requirements include Dash and OR-Tools for the baseline forecasting and replenishment workflow, but intentionally exclude the training stack (Prophet, LightGBM, PyTorch Forecasting, and Darts). Advanced model experiments run in a full local environment. If the deployed service does not download Git LFS objects, the repository-sample button will show an actionable data-loading error; the manual CSV upload remains available.
