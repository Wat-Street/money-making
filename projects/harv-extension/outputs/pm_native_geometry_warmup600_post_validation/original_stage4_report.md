# Stage 4 PM-Native Warmup-600 Results Report

Source artifact: `outputs/pm_native_geometry_warmup600_actions/28733449635_combined`  
GitHub Actions run: https://github.com/Harman6139/money-making/actions/runs/28733449635

## Conclusion

No PM-native construction beat corrected `CP_REPO_FRESH` globally on all four requested metrics (`SMAPE`, `MAE`, `MSE`, `RMSE`).

- `PM_QDK_2` is the strongest PM-native result for SMAPE: +1.478553 CP-minus-model SMAPE over 367,468 rows, positive on 9 of 10 assets. It does not beat CP on MAE, MSE, or RMSE globally.
- `PHQO` is effectively CP-nested and very close to CP. It shows tiny positive pooled advantages on MAE, MSE, RMSE, and SMAPE, but the equal-weight asset SMAPE advantage is slightly negative because some assets have fewer PHQO aligned rows.
- `PM_QDK` loses globally and is not a viable global challenger in this run.

Audit validity is clean: aggregate asset presence, metadata completion, empty failures, repo-CP reproduction, fast-vs-slow, no-lookahead, and global level-sum checks all passed.

## Global Loss Summary

| model_name | loss_metric | n_obs_total | pooled_advantage_CP_minus_model | assets_positive |
| --- | --- | --- | --- | --- |
| PHQO | MAE | 356403 | 4.91088e-08 | 7 |
| PM_QDK_2 | MAE | 367468 | -1.50076e-05 | 7 |
| PM_QDK | MAE | 367468 | -3.01564e-05 | 1 |
| PHQO | MSE | 356403 | 5.88317e-11 | 7 |
| PM_QDK | MSE | 367468 | -2.51049e-07 | 0 |
| PM_QDK_2 | MSE | 367468 | -3.55545e-07 | 0 |
| PHQO | RMSE | 356403 | 1.59185e-08 | 7 |
| PM_QDK | RMSE | 367468 | -7.79766e-05 | 0 |
| PM_QDK_2 | RMSE | 367468 | -0.000123923 | 0 |
| PM_QDK_2 | SMAPE | 367468 | 1.47855 | 9 |
| PHQO | SMAPE | 356403 | 0.0155552 | 8 |
| PM_QDK | SMAPE | 367468 | -1.73763 | 0 |

## Asset Consistency

| model_name | benchmark_model | n_assets | assets_positive_smape_advantage | assets_positive_mae_advantage | equal_weight_mean_smape_advantage | equal_weight_mean_abs_error_advantage |
| --- | --- | --- | --- | --- | --- | --- |
| PM_QDK | CP_REPO_FRESH | 10 | 0 | 1 | -1.73774 | -3.0157e-05 |
| PM_QDK_2 | CP_REPO_FRESH | 10 | 9 | 7 | 1.47808 | -1.50249e-05 |
| PHQO | CP_REPO_FRESH | 10 | 6 | 7 | -0.000603082 | 4.74353e-08 |

## PM_QDK_2 Asset Rows

| asset | n_obs | mean_advantage_vs_CP | mean_abs_error_advantage_vs_CP |
| --- | --- | --- | --- |
| AAPL | 36739 | 1.20705 | 2.83843e-05 |
| AMZN | 36739 | 1.18118 | 3.64741e-05 |
| EEM | 36739 | 1.02756 | -7.86206e-05 |
| FXI | 36817 | 3.71259 | 6.65269e-05 |
| GLD | 36739 | -0.0806828 | -1.87939e-05 |
| GOOGL | 36739 | 0.799278 | 2.91257e-05 |
| HYG | 36739 | 0.452536 | -0.00028501 |
| QQQ | 36739 | 3.20126 | 3.81581e-05 |
| SPY | 36739 | 2.56695 | 1.76014e-05 |
| TLT | 36739 | 0.713076 | 1.59043e-05 |

## PHQO Asset Rows

| asset | n_obs | mean_advantage_vs_CP | mean_abs_error_advantage_vs_CP |
| --- | --- | --- | --- |
| AAPL | 36739 | -3.43549e-15 | 9.3992e-08 |
| AMZN | 36739 | 1.16152e-15 | 1.16944e-07 |
| EEM | 34700 | 0.00265792 | -6.40494e-08 |
| FXI | 36817 | 2.52412e-15 | 1.25405e-07 |
| GLD | 36541 | -0.000924023 | -2.19585e-08 |
| GOOGL | 36739 | -0.0054681 | 1.26942e-07 |
| HYG | 27911 | 0.000356272 | 7.86711e-09 |
| QQQ | 36739 | 4.78829e-15 | 6.30573e-08 |
| SPY | 36739 | 1.1278e-15 | 3.74978e-08 |
| TLT | 36739 | -0.00265288 | -1.13446e-08 |

## Generated Artifacts

- `paper_tables/pm_native_loss_summary.csv`
- `paper_tables/pm_native_asset_consistency.csv`
- `results/pm_native_results_report.md`
- `figures/pm_native_metric_advantages.png`
- `results/aggregate_audit_summary.csv`
