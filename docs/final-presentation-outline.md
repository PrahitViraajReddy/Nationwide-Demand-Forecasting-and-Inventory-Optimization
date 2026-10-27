# Final Presentation Outline (10 Minutes)

Use measured results from the final reproducible run. Do not fill result slides with estimates or invented accuracy/savings.

| Slide | Time | Content |
|---|---:|---|
| 1. Problem and objective | 0:45 | Retail demand variability, stockout/waste trade-off, project objective |
| 2. Dataset and hierarchy | 1:00 | M5 sales/calendar/prices; item-store bottom level; store/state roll-ups; known data limitations |
| 3. Architecture | 1:00 | Ingestion, leakage-safe features, candidate models, evaluation, reconciliation, inventory optimizer, Dash UI |
| 4. Baseline and validation design | 1:00 | Weekly seasonal naive; chronological cutoff; exact series count and horizon; WAPE/RMSSE/MAE definitions |
| 5. Model comparison | 1:30 | Measured baseline, Prophet, LightGBM, and TFT results; runtime and interval coverage/width; explain where a complex model did not win |
| 6. Hierarchical consistency | 0:45 | Before/after aggregation checks; reconciliation method; accuracy and coherence trade-off |
| 7. Inventory decision | 1:15 | Demonstrate on-hand stock, lead time, review period, holding/stockout costs, order cap/multiple, recommendation, modeled cost |
| 8. Live demo | 1:00 | Load M5 sample or CSV, generate forecast, inspect holdout metrics, optimize replenishment, download output |
| 9. Challenges and ownership | 0:45 | Data scale, LFS/deployment, leakage prevention, solver assumptions, what you personally implemented |
| 10. Limitations and next steps | 0:20 | Untouched test results, full-hierarchy scale, operational inputs, monitoring and deployment |

## Before presenting

- [ ] Run the same experiment configuration twice and verify reproducible metrics.
- [ ] Record commit SHA, Python/dependency versions, seed, data file, cutoff, and horizon.
- [ ] Save forecast CSVs and metrics JSON in the local artifacts directory; commit only small, reviewed summary tables if appropriate.
- [ ] Capture screenshots or a short recording of the dashboard with a known sample.
- [ ] Clearly label any demo cost assumptions and avoid calling modeled cost reduction realized savings.
- [ ] Prepare one example where the baseline is competitive and explain why model complexity is or is not justified.
