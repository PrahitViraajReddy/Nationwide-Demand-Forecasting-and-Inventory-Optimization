# Architecture and Design Decisions

## End-to-end flow

1. **Ingestion and validation** — read local M5 sales, calendar, and sell-price tables; validate expected files, keys, date coverage, and joins.
2. **Model-ready dataset** — transform wide daily sales into date-indexed series and join calendar/price features at the correct grain. Keep raw files immutable.
3. **Temporal validation** — define chronological train/validation/test cutoffs; all lags and rolling features use past observations only.
4. **Forecast candidates** — seasonal-naive baseline, Prophet baseline, LightGBM global model, and a bounded Temporal Fusion Transformer experiment.
5. **Evaluation and uncertainty** — calculate WAPE/RMSSE by item-store and aggregate level; assess prediction-interval coverage and width.
6. **Hierarchy reconciliation** — use Darts for reconciliation experiments and verify that child forecasts sum to parent totals within numerical tolerance.
7. **Inventory optimization** — OR-Tools converts demand distributions into order quantities using configurable lead time, service target, on-hand inventory, order limits, and cost assumptions.
8. **Decision interface** — Plotly Dash presents filters, forecast bands, actual-versus-predicted plots, metrics, and replenishment recommendations.

## Logical components

- `demand_inventory.data`: dataset paths and data integrity checks.
- `demand_inventory.metrics`: evaluation metrics.
- Forecasting modules (planned): baselines, feature generation, LightGBM, Prophet, TFT, and reconciliation.
- Inventory module (planned): decision variables, objective, constraints, and solver diagnostics.
- Dash application (planned): presentation layer calling the same service functions as tests and scripts.

## Core design principles

- Keep data preparation, modeling, optimization, and UI separate.
- Make cutoffs, horizons, model settings, and inventory assumptions configurable.
- Never fit preprocessing or use future observations before the forecast origin.
- Preserve a baseline and compare every more complex model against it.
- Persist experiment configuration and results so reported numbers are reproducible.
- Treat unavailable operational values as explicit assumptions—not facts supplied by M5.

## Inventory objective (initial formulation)

Minimize expected shortage penalty plus holding cost, subject to nonnegative order quantities and any configured order/capacity constraints. The demand distribution, review period, lead time, service target, unit costs, and current inventory are inputs. The implementation must document how these inputs are estimated or assumed.

## Scale strategy

Develop and test on a small, fixed subset first. Run full-hierarchy training and TFT only after measuring memory/runtime. The complete M5 tables are large; generated features, checkpoints, and raw data are excluded from version control.
