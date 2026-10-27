# M5 Data Dictionary

The original M5 competition CSVs are tracked as Git LFS objects at the repository root. The loader also accepts a local copy under `data/raw/m5/`. Run `git lfs install` and `git lfs pull` before accessing the dataset. This document describes fields used by the current data loader and forecasting experiments.

## Sales table

- `id`: series identifier for an item at a store and evaluation suffix.
- `item_id`, `dept_id`, `cat_id`: product hierarchy identifiers.
- `store_id`, `state_id`: store and geographic hierarchy identifiers.
- `d_1`, `d_2`, ...: daily unit sales observations in the competition's day-index format.

## Calendar table

- `d`: day key used by sales columns.
- `date`: calendar date.
- `wm_yr_wk`: week key used for price joins.
- Event and SNAP fields: calendar/event context and state-specific SNAP indicators; use only information available at forecast time.

## Sell-price table

- `store_id`, `item_id`, `wm_yr_wk`: join keys.
- `sell_price`: weekly selling price for that item/store combination.

## Join and leakage rules

- Map each `d_*` column to `calendar.d` before creating a date-indexed series.
- Join prices on `(store_id, item_id, wm_yr_wk)`, not item alone.
- Handle absent price rows as missing values in LightGBM preparation; the current TFT preparation fills missing prices with `0.0`, so interpret that as a missing-price sentinel and consider adding a separate price-availability feature before final experiments.
- Keep identifiers categorical where appropriate; do not treat arbitrary IDs as continuous quantities without justification.
- Event fields and prices in the validation horizon are usable only if they would have been known at forecast time.
- The loader prefers `sales_train_validation.csv` when both sales files exist, otherwise it uses `sales_train_evaluation.csv`. Record the chosen file in experiment metadata before publishing benchmark results.
