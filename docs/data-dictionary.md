# M5 Data Dictionary

The project expects the original M5 competition tables; this document describes the fields used by the planned workflow.

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
- Handle absent price rows as missing values, not automatically as zero price.
- Keep identifiers categorical where appropriate; do not treat arbitrary IDs as continuous quantities without justification.
- Event fields and prices in the validation horizon are usable only if they would have been known at forecast time.
- Document the exact sales file used (`validation` versus `evaluation`) in every experiment.
