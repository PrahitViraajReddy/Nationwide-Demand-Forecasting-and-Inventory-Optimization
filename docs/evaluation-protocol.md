# Evaluation Protocol

## Evaluation windows

- Split by date, never by random rows.
- Reserve the final horizon for validation during development; retain a separate untouched test window for final reporting.
- Compare every candidate against the weekly seasonal-naive baseline on the same series and dates.
- Fit feature encoders and model parameters on training data only. Recursive models must feed their own predictions into later lags after the forecast origin.
- Report the exact sales file, series-selection rule, cutoff dates, horizon, series count, aggregation level, seed, and model configuration with every metric.
- Do not publish development-subset metrics as full-M5 leaderboard performance.

## Forecast metrics

For actual demand (y_t) and forecast (hat y_t):

- **WAPE** = (sum_t |y_t-hat y_t| / sum_t |y_t|). It is undefined when total absolute actual demand is zero.
- **MAE** = mean (|y_t-hat y_t|), in units sold.
- **RMSSE** = square root of mean squared forecast error divided by the training-history mean squared seasonal-naive difference. Use seasonal period 7 for daily demand when evaluating against a weekly seasonal pattern.
- **Interval coverage** = fraction of actuals inside the reported lower/upper bounds.
- **Mean interval width** = average upper minus lower bound. Coverage must be interpreted together with width; wide intervals can achieve high coverage without being useful.

Report WAPE and RMSSE at both bottom level and aggregate levels where feasible. RMSSE aggregation must be labeled clearly (for example, mean per-series RMSSE versus RMSSE computed on an aggregate series).

## Inventory evaluation

The current OR-Tools formulation minimizes modeled holding cost on projected ending inventory plus a stockout penalty on unmet forecast demand. It is a deterministic single-period planning approximation.

At minimum, report:

- forecast demand over the lead-time + review-period window;
- starting on-hand inventory and all assumed operational parameters;
- recommended order quantity and any order-cap/multiple constraint;
- projected ending inventory and unmet demand;
- modeled holding, shortage, and total cost;
- whether the solver returned an optimal solution.

The M5 dataset does not supply on-hand stock, true lead times, supplier constraints, or business cost parameters. Treat dashboard inputs as illustrative assumptions until operational data is available. Do not claim realized savings from modeled cost alone.

## Minimum comparison table

| Model | Series | Horizon | Validation dates | WAPE | RMSSE | Interval coverage | Mean interval width | Runtime |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Weekly seasonal naive | | | | | | N/A | N/A | |
| Prophet | | | | | | | | |
| LightGBM | | | | | | | | |
| Temporal Fusion Transformer | | | | | | | | |

Leave cells blank until a reproducible experiment produces the result. Record failed runs and their causes instead of silently excluding them.
