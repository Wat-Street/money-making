# CP_REPO_OPS_HG 18-Model Audit Report

Run: `28269511971`
Repo/branch: `Harman6139/money-making`, `cp-pm-incremental-actions`
Corrected runner: `projects/harv-extension/outputs/cp_repo_ops_hg_tests/scripts/run_cp_repo_ops_hg_tests.py`
Artifact family: `projects/harv-extension/outputs/cp_repo_ops_hg_actions/28269511971/`

## Bottom Line

The CP benchmark and the broad underperformance result are directionally valid, but the run is not clean enough for final paper claims without tightening the audits and fixing one unstable model.

No fatal bug was found in `CP_REPO_FRESH`, target alignment, benchmark reproduction, feature purity, or ridge penalty placement. The corrected benchmark is the repo `contig_prime_modulo` CP design plus repo RV/HAR columns, not the old four-block `CP_FRESH` design.

Two defects matter:

1. `manual_model_conditional_summary.csv` has a real metadata bug: `n_assets` is `1` on every conditional row, while recomputation from the ten artifacts gives `10`. Performance columns match aside from floating point noise.
2. `CP_REPO_LRPM` is numerically unstable: it is unregularized, rank deficient on every asset, and produces explosive predictions on multiple assets. That model's exact loss magnitude is not a credible model-quality measurement until the LRPM design is stabilized or ridge-regularized.

Blunt conclusion: results are directionally valid but audits need strengthening before final paper claims.

## A. Run And Artifact Verification

GitHub Actions run `28269511971` completed with overall conclusion `failure`, but this was caused by the aggregate artifacts job after the asset outputs had already been uploaded.

Confirmed from the run metadata:

- All 10 asset jobs completed successfully: AAPL, AMZN, EEM, FXI, GLD, GOOGL, HYG, QQQ, SPY, TLT.
- Each asset job completed `Run full ladder for asset`, `Validate asset output`, and `Upload asset artifact` successfully.
- The aggregate job succeeded through `Download per-asset artifacts` and `Combine per-asset outputs`; it failed/cancelled during `Regenerate combined tables`.
- GitHub lists exactly 10 non-expired artifacts, one per asset.
- I downloaded the artifacts to `downloaded_artifacts/` and verified every required file listed in the request exists. Missing count: `0`.

## B. Manual Summary Recompute

Recomputed from the ten per-asset artifact tables, not from the manual summary.

Manual overall comparison:

- `manual_model_overall_summary.csv` matches recomputation except one floating point roundoff row: `CP_REPO_LRPM`, `pooled_mean_abs_error_model`, absolute diff `3.725290e-09`. That value is far below meaningful scale, though it is greater than the strict `1e-10` threshold because the number itself is huge.

Manual conditional comparison:

- `manual_model_conditional_summary.csv` has `296` diffs beyond `1e-10`.
- `288` are the same real bug: `n_assets` is `1` manually and `10` when recomputed.
- The other `8` diffs are floating point noise in LRPM huge absolute-error columns, max `1.490116e-08`.
- Conditional performance metrics otherwise match.

Recomputed files:

- `audit_recomputed/recomputed_model_overall_summary.csv`
- `audit_recomputed/recomputed_model_conditional_summary.csv`
- `audit_recomputed/manual_overall_diffs_gt_1e-10.csv`
- `audit_recomputed/manual_conditional_diffs_gt_1e-10.csv`

Overall SMAPE ranking:

| model_name | pooled_mean_advantage_vs_CP | equal_weight_asset_mean_advantage_vs_CP | pooled_win_rate_vs_CP | assets_positive | assets_negative |
| --- | --- | --- | --- | --- | --- |
| CP_REPO_FRESH | 0 | 0 | 0 | 0 | 0 |
| CP_REPO_OPS_K | -0.0853676 | -0.0853131 | 0.419514 | 0 | 10 |
| CP_REPO_RIDGE_OPS_R | -0.135219 | -0.135162 | 0.412354 | 1 | 9 |
| CP_REPO_OPS_HG | -0.144797 | -0.144556 | 0.411848 | 4 | 6 |
| RANDOM_RESIDUES_PLACEBO_REPO | -0.154396 | -0.154245 | 0.418107 | 2 | 8 |
| CP_REPO_GATED_RIDGE_OPS_C | -0.159117 | -0.15901 | 0.418091 | 1 | 9 |
| RANDOM_GATE_PLACEBO_REPO | -0.198547 | -0.19846 | 0.409943 | 0 | 10 |
| SHUFFLED_LAG_PM_PLACEBO_REPO | -0.201074 | -0.201016 | 0.414749 | 1 | 9 |
| CP_REPO_RECENT_SLOPE | -0.212829 | -0.212644 | 0.413315 | 4 | 6 |
| RAW_CP_REPO_PLUS_PM | -0.224482 | -0.224337 | 0.407494 | 1 | 9 |
| CP_REPO_HAAR_SHAPE | -0.264459 | -0.264319 | 0.410297 | 0 | 10 |
| CP_REPO_RIDGE_OPS_C | -0.269414 | -0.269274 | 0.410898 | 1 | 9 |
| RIDGE_AR22 | -0.503013 | -0.503052 | 0.409184 | 2 | 8 |
| CP_REPO_OPS_C | -0.619446 | -0.619214 | 0.408027 | 0 | 10 |
| RAW_PM | -1.02741 | -1.02753 | 0.401368 | 1 | 9 |
| HAR_RV | -1.29013 | -1.2903 | 0.381494 | 1 | 9 |
| CP_REPO_OPS_R | -7.96502 | -7.96436 | 0.363585 | 0 | 10 |
| CP_REPO_LRPM | -12.1427 | -12.1419 | 0.343219 | 0 | 10 |

Every always-on challenger has negative pooled SMAPE advantage versus `CP_REPO_FRESH`. The best challenger is `CP_REPO_OPS_K` at `-0.0853676` pooled SMAPE points with `0/10` positive assets.

## C. CP Benchmark Correctness

Code audit:

- `build_feature_frame` calls `contig_prime_modulo(vol.copy(), n, per_day_normalize=False)`.
- `CP_REPO_FRESH` uses `cp_cols = sorted([col for col in frame.columns if col.startswith("RV") or col.startswith("CP_")])`.
- `CPB_B*` starts with `CPB`, not `CP_`, so four-block diagnostic columns are excluded.
- PM columns, OPS columns, shape columns, gates, and placebos are excluded by construction and by `validate_model_features`.

Repo CP reproduction audit across all 10 assets:

- All rows have `passes = True`.
- Actual targets match exactly within `1e-12`.
- Mean absolute CP reproduction diff range: `3.07992e-13` to `2.35557e-12`.
- Max absolute diff across all assets: `1.11383e-09`.
- The max diff is small numerical reproduction noise from CSV/linear algebra ordering and is below the runner's `1e-7` pass threshold.

Independent spot check from saved old `Predicted_CP` versus artifact `Predicted_CP_REPO_FRESH`:

| asset | n_saved | n_recomputed | n_aligned | actual_match | mean_abs_diff | median_abs_diff | max_abs_diff | first_aligned_date | last_aligned_date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AAPL | 36739 | 36739 | 36739 | True | 7.5518e-13 | 1.243e-13 | 2.51978e-10 | 2023-08-17 16:00:00 | 2025-07-07 16:00:00 |
| QQQ | 36739 | 36739 | 36739 | True | 1.07457e-12 | 3.224e-13 | 2.3001e-10 | 2023-08-17 16:00:00 | 2025-07-07 16:00:00 |
| SPY | 36739 | 36739 | 36739 | True | 1.65099e-12 | 5.004e-13 | 2.59374e-10 | 2023-08-17 16:00:00 | 2025-07-07 16:00:00 |

Conclusion: `CP_REPO_FRESH` correctly reproduces the old repo CP predictions. The benchmark is not accidentally the old four-block CP benchmark.

## D. Forecast Alignment And Target Timing

The code follows the expected convention:

- `y_next = frame["RV_d"].shift(-1)`.
- For forecast origin `i`, `fast_expanding_predict` has accumulated training feature rows `< i`.
- Prediction uses feature row `i`.
- Output timestamp is `frame.index[i + 1]`.
- The target is `RV_d` at row `i + 1`.

AAPL first forecast timeline:

| asset | n | warmup | first_forecast_origin_i | train_feature_rows | train_feature_timestamp_start | train_feature_timestamp_end | train_target_rows | train_target_timestamp_start | train_target_timestamp_end | feature_row_i | feature_timestamp | target_row_i_plus_1 | target_timestamp | target_RV_d | prediction_file_first_date | prediction_file_first_actual | prediction_file_first_cp_pred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AAPL | 22 | 600 | 622 | 0..621 | 2023-08-08 09:35:00 | 2023-08-17 15:50:00 | 1..622 | 2023-08-08 09:40:00 | 2023-08-17 15:55:00 | 622 | 2023-08-17 15:55:00 | 623 | 2023-08-17 16:00:00 | 0.000382023 | 2023-08-17 16:00:00 | 0.000382023 | 0.00113187 |

This proves the first output date `2023-08-17 16:00:00` uses feature row `2023-08-17 15:55:00`; training feature rows end at `2023-08-17 15:50:00`, and training targets end at `2023-08-17 15:55:00`. The model does not see the target row it is predicting.

The included no-lookahead audit also passed on all assets, but it is synthetic and should be treated as a smoke test, not a substitute for this timeline check.

## E. Fast-Vs-Slow Equivalence

The original artifact audit is too weak:

- It checks only 9 selected models.
- It uses only 8 forecasts via `max_forecasts=8`.
- It does not prove all 18 models or full-run equivalence.

I implemented a stronger audit in `audit_recomputed/strong_fast_slow_equivalence_audit.csv`:

- All 18 models.
- 500 forecasts per model.
- Assets: AAPL, SPY, GLD.
- Same selected lambdas as the run metadata.
- Same features, target transform, penalty matrix, timestamps, and actuals.
- Thresholds: `1e-8` full rank, `1e-6` rank deficient.

Strong audit pass summary:

| asset | sum | count |
| --- | --- | --- |
| AAPL | 18 | 18 |
| GLD | 18 | 18 |
| SPY | 18 | 18 |

All 54 model/asset checks passed. This materially strengthens the original audit, but a full-AAPL all-row slow audit was not run because it is much more expensive.

## F. Feature Purity

`validate_model_features` enforces the intended model families. Rebuilt feature groups produced no feature-purity failures:

- CPB columns entering paper models: `0` failures.
- Raw PM leaking into cleaned models: `0` failures.
- Ungated OPS/shape leaking into gated models: `0` failures.
- True OPS leaking into placebo models: `0` failures.
- CP controls omitted from `CP_REPO_*` models: `0` failures.

`CP_REPO_OPS_K` is not a placeholder in this run. It has 21 features: 13 repo CP/RV controls plus 8 generated gated KOPS features. It is marked paper-eligible in metadata, though it is still just an optional kernel challenger, not proof of PM-specific structure.

## G. Ridge And Partial-Ridge Math

Ridge audit file: `audit_recomputed/ridge_penalty_audit.csv`.

Findings:

- Intercept is unpenalized by explicit `penalty_matrix[0, 0] = 0.0`.
- Repo RV/CP controls are unpenalized in every ridge model.
- Shape, PM, OPS, gated, placebo, or AR lag features are penalized in ridge models.
- Ridge audit failures: `0`.
- `CP_REPO_OPS_HG` applies `lambda_C` to gated OPS-C and `lambda_R = ratio * lambda_C` to gated OPS-R.
- Ratio grid included `2, 5, 10, 20`.
- Selected HG ratios:

| selected_lambda_r_ratio | asset_count |
| --- | --- |
| 2 | 5 |
| 5 | 1 |
| 20 | 4 |

This satisfies the partial-ridge math: the objective penalizes only `theta` on extra features, not the intercept or CP controls.

## H. Hyperparameter Selection

`cv-mode inner` was used in the full run for all ridge-family models. Selection is per asset/model and happens before OOS forecasts. The selected lambda is then held fixed for all OOS forecasts.

Important limitations:

- The validation window is small: with `n=22`, `warmup=600`, first forecast origin is index `622`; validation is roughly rows `497..621`, about 125 rows.
- Validation optimizes SMAPE, while model fitting optimizes squared error with ridge penalties.
- The grid is primitive: `0.0001, 0.001, 0.01`, with HG ratios `2, 5, 10, 20`.
- Many models select the upper edge `0.01`, especially OPS-C/HG/K variants. That means the grid is probably too narrow.
- I found no OOS leakage in lambda selection and no fallback rows in metadata.

Selected lambda shape counts:

| model_name | 0.0001 | 0.001 | 0.01 |
| --- | --- | --- | --- |
| CP_REPO_GATED_RIDGE_OPS_C | 1 | 0 | 9 |
| CP_REPO_OPS_HG | 2 | 0 | 8 |
| CP_REPO_OPS_K | 1 | 0 | 9 |
| CP_REPO_RIDGE_OPS_C | 0 | 0 | 10 |
| CP_REPO_RIDGE_OPS_R | 3 | 0 | 7 |
| RANDOM_GATE_PLACEBO_REPO | 1 | 0 | 9 |
| RANDOM_RESIDUES_PLACEBO_REPO | 6 | 0 | 4 |
| RIDGE_AR22 | 3 | 3 | 4 |
| SHUFFLED_LAG_PM_PLACEBO_REPO | 2 | 0 | 8 |

This is good enough for an audit run, not good enough for final hyperparameter claims.

## I. Gate Logic

Gate construction uses only lag columns:

- Recent slope: mean lags 1-3 minus mean lags 4-8.
- Spike recency: position of max lag value inside the lag vector.
- Exit pressure: older block mean minus recent block mean.
- Dispersion: older-lag dispersion.
- Components are online-standardized with expanding mean/std shifted by 1 row.
- `GATE_RANDOM` is a permutation of `GATE_REAL`, preserving the exact empirical distribution.

Gate and conditional advantage summary:

| condition | n_obs_total | n_assets | mean_real_gate | mean_random_gate | raw_PM_advantage | gated_OPS_C_advantage | OPS_HG_advantage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster_entry_loose | 73457 | 10 | 0.736194 | 0.447549 | -0.477918 | -0.164951 | 0.192061 |
| recent_spike_position | 125358 | 10 | 0.637924 | 0.446977 | -0.30453 | 0.0077864 | 0.229661 |
| cluster_exit_loose | 73455 | 10 | 0.179418 | 0.447099 | -2.09539 | -0.271577 | -0.530026 |
| older_spike_position | 108871 | 10 | 0.289241 | 0.445845 | -1.44631 | -0.16639 | -0.280839 |
| high_within_block_dispersion | 68970 | 10 | 0.437889 | 0.44594 | -2.72077 | -0.0293911 | 0.560382 |
| entry_with_recent_spike | 43747 | 10 | 0.787351 | 0.447311 | 0.0341606 | 0.144437 | 0.585652 |
| stable_low_dispersion | 68968 | 10 | 0.407417 | 0.444945 | -0.105876 | -0.202122 | -0.480828 |
| all_observations | 367248 | 10 | 0.446605 | 0.446586 | -1.02395 | -0.154198 | -0.136635 |

Interpretation:

- The real gate separates regimes: entry/recent-spike rows have high gate values; exit/older-spike rows have low gate values.
- Random gate stays near the unconditional mean around `0.446`, as expected.
- `CP_REPO_OPS_HG` is conditionally useful: `+0.586` SMAPE points on `entry_with_recent_spike`, `+1.879` on `recent_ramp_high_dispersion`, and `+0.560` on high dispersion.
- It is harmful in exits, older spikes, and stable low dispersion.
- Gating helps, but always-on gated linear features still fire too broadly.

## J. Condition Definitions

Conditions are computed from lagged realized targets in `add_lagged_actuals` and `define_conditions` after predictions are aligned. They do not use the current target being evaluated or future targets.

Caveats:

- Quantiles are computed on the full evaluation frame. That is acceptable for ex-post descriptive labels but not for tradable online labels.
- Conditions are not mutually exclusive. Pooled conditional effects are descriptive, not additive.
- Conditional `all_observations` excludes the first 22 prediction rows per asset because lagged condition features are required. That is why conditional total is `367248`, not overall total `367468`.

## K. Mathematical Explanation For Underperformance

Let `C_t` be repo CP/RV controls and `Z_t` extra PM/shape features. Define `delta_t = yhat_M - yhat_CP` and CP residual `e_t = y_t - yhat_CP`.

For squared loss, challenger improvement is approximately:

```text
L_CP - L_M = 2 e_t delta_t - delta_t^2
```

The residual perturbation diagnostics are in `audit_recomputed/residual_perturbation_diagnostics.csv`.

Overall residual perturbation summary:

| model_name | mean_2e_delta_pooled | mean_delta_squared_pooled | ratio_2Eedelta_over_Edelta2_pooled | actual_smape_advantage_pooled | win_rate_vs_cp_pooled | assets_positive_smape_advantage |
| --- | --- | --- | --- | --- | --- | --- |
| CP_REPO_OPS_K | 2.97295e-09 | 2.52146e-09 | 1.17906 | -0.0853676 | 0.419514 | 0 |
| CP_REPO_RIDGE_OPS_R | 1.99415e-09 | 1.72364e-09 | 1.15694 | -0.135219 | 0.412354 | 1 |
| CP_REPO_OPS_HG | 3.8205e-08 | 1.99109e-08 | 1.9188 | -0.144797 | 0.411848 | 4 |
| RANDOM_RESIDUES_PLACEBO_REPO | 1.83645e-08 | 9.3471e-09 | 1.96473 | -0.154396 | 0.418107 | 2 |
| CP_REPO_GATED_RIDGE_OPS_C | 2.00211e-08 | 1.14056e-08 | 1.75537 | -0.159117 | 0.418091 | 1 |
| RANDOM_GATE_PLACEBO_REPO | 1.68954e-08 | 1.04399e-08 | 1.61835 | -0.198547 | 0.409943 | 0 |
| SHUFFLED_LAG_PM_PLACEBO_REPO | 1.18661e-08 | 6.45922e-09 | 1.83709 | -0.201074 | 0.414749 | 1 |
| CP_REPO_RECENT_SLOPE | 9.73107e-09 | 4.78563e-09 | 2.03339 | -0.212829 | 0.413315 | 4 |
| RAW_CP_REPO_PLUS_PM | 2.29846e-08 | 1.17523e-08 | 1.95576 | -0.224482 | 0.407494 | 1 |
| CP_REPO_HAAR_SHAPE | 3.14662e-08 | 1.55639e-08 | 2.02174 | -0.264459 | 0.410297 | 0 |
| CP_REPO_RIDGE_OPS_C | 3.17293e-08 | 1.54125e-08 | 2.05867 | -0.269414 | 0.410898 | 1 |
| RIDGE_AR22 | 3.11243e-08 | 7.05574e-08 | 0.441121 | -0.503013 | 0.409184 | 2 |
| CP_REPO_OPS_C | 7.95027e-08 | 2.66756e-05 | 0.00298035 | -0.619446 | 0.408027 | 0 |
| RAW_PM | 1.36963e-08 | 6.20812e-08 | 0.220619 | -1.02741 | 0.401368 | 1 |
| HAR_RV | 3.61676e-11 | 4.58917e-08 | 0.000788109 | -1.29013 | 0.381494 | 1 |
| CP_REPO_OPS_R | -3.51668e-06 | 0.384317 | -9.15048e-06 | -7.96493 | 0.363631 | 0 |
| CP_REPO_LRPM | -30439.7 | 1.14678e+20 | -2.65436e-16 | -12.1428 | 0.343239 | 0 |

The important nuance: several challengers improve squared-error-like losses, even though they lose under SMAPE. For example, `CP_REPO_OPS_HG` has `ratio_2Eedelta_over_Edelta2 = 1.9188`, which is favorable under squared loss, but its SMAPE advantage is `-0.1448`.

So the correct explanation is not simply "the residual signal is absent." The sharper explanation is:

1. CP captures the dominant volatility persistence component.
2. Extra PM/shape features create perturbations with real signal in entry/recent-spike regimes.
3. The perturbations are harmful in exit/older-spike/stable/noisy regimes.
4. SMAPE weights relative errors and near-zero-volatility behavior aggressively, so small bad adjustments can dominate broad loss even when MAE/MSE improve.
5. Win rates are below 42% for all always-on challengers, so gains are event-size/gain-size effects, not broad row-level dominance.
6. Residualized feature signal is weak. Mean absolute correlation of residualized extra features with CP residual is small for PM-specific OPS-R (`~0.0087`) and modest even for the best generic shape blocks.
7. Placebos are close to real PM variants, so PM-specific residue structure is weakly supported at best.

Residualized extra-feature signal:

| model_name | n_assets | mean_r2 | mean_abs_corr | max_abs_corr |
| --- | --- | --- | --- | --- |
| CP_REPO_RECENT_SLOPE | 10 | 0.171985 | 0.0212502 | 0.0728995 |
| RIDGE_AR22 | 10 | 0.192121 | 0.0179592 | 0.0861293 |
| RANDOM_RESIDUES_PLACEBO_REPO | 10 | 0.0971568 | 0.0175709 | 0.0473644 |
| SHUFFLED_LAG_PM_PLACEBO_REPO | 10 | 0.0651706 | 0.0173047 | 0.0440042 |
| CP_REPO_RIDGE_OPS_C | 10 | 0.0744676 | 0.0172518 | 0.0923881 |
| CP_REPO_OPS_C | 10 | 0.0744676 | 0.0172518 | 0.0923881 |
| CP_REPO_GATED_RIDGE_OPS_C | 10 | 0.0467636 | 0.0164881 | 0.0675656 |
| CP_REPO_OPS_HG | 10 | 0.0697945 | 0.0152052 | 0.0675656 |
| RANDOM_GATE_PLACEBO_REPO | 10 | 0.0478934 | 0.0138059 | 0.063671 |
| CP_REPO_HAAR_SHAPE | 10 | 0.0669953 | 0.0123006 | 0.0772026 |
| RAW_CP_REPO_PLUS_PM | 10 | 0.565044 | 0.0102135 | 0.0324556 |
| RAW_PM | 10 | 0.565044 | 0.0102135 | 0.0324556 |
| CP_REPO_OPS_K | 10 | 0.0678477 | 0.00924941 | 0.0365692 |
| CP_REPO_OPS_R | 10 | 0.073818 | 0.00873427 | 0.0258258 |
| CP_REPO_RIDGE_OPS_R | 10 | 0.073818 | 0.00873427 | 0.0258258 |
| CP_REPO_LRPM | 10 | 0.0780787 | 0.00794028 | 0.0259509 |

## L. Alternative Losses

Alternative loss file: `audit_recomputed/alternative_loss_summary.csv`.

Overall standard-loss advantages, CP minus model:

| model_name | MAE | MSE | RMSE | SMAPE | log_RV_MSE |
| --- | --- | --- | --- | --- | --- |
| CP_REPO_OPS_K | -4.84538e-08 | 4.51494e-10 | 5.25263e-08 | -0.0853676 | -0.0807576 |
| CP_REPO_RIDGE_OPS_R | -6.29527e-07 | 2.70509e-10 | 8.19013e-08 | -0.135219 | -0.142895 |
| CP_REPO_OPS_HG | 4.71426e-06 | 1.82941e-08 | 5.47021e-06 | -0.144797 | 0.0331629 |
| RANDOM_RESIDUES_PLACEBO_REPO | 2.10319e-06 | 9.01741e-09 | 2.56238e-06 | -0.154396 | -0.186958 |
| CP_REPO_GATED_RIDGE_OPS_C | 1.10633e-06 | 8.61548e-09 | 2.37108e-06 | -0.159117 | -0.0498523 |
| RANDOM_GATE_PLACEBO_REPO | 3.90318e-07 | 6.4555e-09 | 1.79823e-06 | -0.198547 | -0.178442 |
| SHUFFLED_LAG_PM_PLACEBO_REPO | 1.31179e-06 | 5.40692e-09 | 1.62005e-06 | -0.201074 | -0.29203 |
| CP_REPO_RECENT_SLOPE | 3.81992e-07 | 4.94544e-09 | 1.65269e-06 | -0.212829 | 0.287391 |
| RAW_CP_REPO_PLUS_PM | 1.07956e-06 | 1.12324e-08 | 3.30549e-06 | -0.224482 | -0.202885 |
| CP_REPO_HAAR_SHAPE | 1.67298e-06 | 1.59023e-08 | 4.49781e-06 | -0.264459 | -0.28136 |
| CP_REPO_RIDGE_OPS_C | 1.63237e-06 | 1.63168e-08 | 4.60049e-06 | -0.269414 | -0.292059 |
| RIDGE_AR22 | -5.93985e-06 | -3.94331e-08 | -1.21921e-05 | -0.503013 | -0.484955 |
| CP_REPO_OPS_C | -2.41038e-05 | -2.65961e-05 | -0.00213625 | -0.619446 | -0.754638 |
| RAW_PM | -1.30482e-05 | -4.83849e-08 | -1.47239e-05 | -1.02741 | -0.731102 |
| HAR_RV | -1.75245e-05 | -4.58555e-08 | -1.4152e-05 | -1.29013 | -1.08269 |
| CP_REPO_OPS_R | -0.00239349 | -0.384272 | -0.241984 | -7.95377 | -11.5409 |
| CP_REPO_LRPM | -2.04931e+07 | -1.14672e+20 | -3.65691e+09 | -12.1387 | -16.9214 |

Key finding: SMAPE is masking useful performance for some models.

- `CP_REPO_OPS_HG` improves MAE by `4.714e-06`, MSE by `1.829e-08`, and RMSE by `5.470e-06`, but loses SMAPE by `0.1448` points.
- `RAW_CP_REPO_PLUS_PM`, `CP_REPO_RIDGE_OPS_C`, `CP_REPO_HAAR_SHAPE`, and several placebos also improve MSE/RMSE while losing SMAPE.
- The research claim therefore depends materially on the loss function.
- QLIKE-like diagnostics are unstable here because realized-volatility targets and some predictions are near zero or clipped; do not lean on QLIKE without a better variance-loss specification.

## M. Selective Overlay Tests

Switching overlay file: `audit_recomputed/switching_overlay_summary.csv`.

The overlay uses CP everywhere, then applies a challenger's perturbation only when a gate/condition is active.

Top switching strategies:

| model_name | strategy | mean_coverage_equal_weight | n_active_total | mean_advantage_active_rows_weighted | mean_advantage_overall_equal_weight | win_rate_overall_equal_weight | false_positive_loss_mean_equal_weight | false_negative_cost_mean_equal_weight | assets_positive_overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CP_REPO_OPS_HG | top_10pct_gate | 0.100003 | 36748 | 1.13622 | 0.11362 | 0.0433071 | 11.3417 | 7.30266 | 9 |
| CP_REPO_OPS_HG | top_5pct_gate | 0.0500016 | 18374 | 2.24593 | 0.112282 | 0.0228591 | 12.704 | 7.46146 | 9 |
| RAW_CP_REPO_PLUS_PM | top_30pct_gate | 0.300007 | 110243 | 0.30178 | 0.0905049 | 0.129333 | 4.34614 | 6.25371 | 9 |
| RAW_CP_REPO_PLUS_PM | top_20pct_gate | 0.200007 | 73496 | 0.452386 | 0.0904519 | 0.0874193 | 4.4424 | 6.00975 | 9 |
| RAW_CP_REPO_PLUS_PM | top_10pct_gate | 0.100003 | 36748 | 0.878655 | 0.0878515 | 0.0447575 | 4.9839 | 5.7708 | 9 |
| RAW_CP_REPO_PLUS_PM | condition_entry_with_recent_spike | 0.119049 | 43747 | 0.667818 | 0.0794962 | 0.0557957 | 5.74766 | 5.67916 | 9 |
| RAW_CP_REPO_PLUS_PM | top_5pct_gate | 0.0500016 | 18374 | 1.51326 | 0.0756476 | 0.0232019 | 5.9347 | 5.67923 | 9 |
| CP_REPO_OPS_HG | top_20pct_gate | 0.200007 | 73496 | 0.36462 | 0.072928 | 0.0835173 | 10.6281 | 6.93683 | 8 |
| CP_REPO_OPS_HG | condition_entry_with_recent_spike | 0.119049 | 43747 | 0.585652 | 0.0697275 | 0.0549769 | 11.7395 | 7.21749 | 9 |
| RAW_CP_REPO_PLUS_PM | condition_cluster_entry_loose | 0.1999 | 73457 | 0.246315 | 0.049242 | 0.0899434 | 5.93156 | 5.6497 | 9 |

This is the strongest research lead in the audit:

- `CP_REPO_OPS_HG` on top 10% gate rows gives `+0.1136` equal-weight overall SMAPE advantage and `+1.136` active-row advantage, positive on `9/10` assets.
- `CP_REPO_OPS_HG` on top 5% gate rows gives `+0.1123` overall and `+2.246` active-row advantage, positive on `9/10` assets.
- `RAW_CP_REPO_PLUS_PM` with top 20-30% gate rows is also positive.

The product implied by the data is not "beat CP everywhere." It is "CP baseline with a rare-regime overlay."

## N. Model-Specific Defect: CP_REPO_LRPM

`CP_REPO_LRPM` is unregularized and rank deficient on every asset:

| asset | feature_count | design_rank | rank_deficient | model_family | selected_lambda_shape |
| --- | --- | --- | --- | --- | --- |
| AAPL | 38 | 33 | True | linear | 0 |
| AMZN | 38 | 33 | True | linear | 0 |
| EEM | 38 | 33 | True | linear | 0 |
| FXI | 38 | 33 | True | linear | 0 |
| GLD | 38 | 33 | True | linear | 0 |
| GOOGL | 38 | 33 | True | linear | 0 |
| HYG | 38 | 33 | True | linear | 0 |
| QQQ | 38 | 33 | True | linear | 0 |
| SPY | 38 | 33 | True | linear | 0 |
| TLT | 38 | 33 | True | linear | 0 |

It produces explosive predictions on multiple assets, including negative predictions below `-1e10` and positive predictions above `1e12` in spot checks. Because SMAPE is bounded, this does not explode the SMAPE table as much as MAE/MSE, but the model is numerically invalid as an unregularized linear challenger.

This does not invalidate `CP_REPO_FRESH`, the summary recomputation, the feature-purity audit, or the better-performing challengers. It does mean LRPM should be fixed or excluded before any final claim about the 18-model ladder.

## Direct Answers

1. Are the current 18-model results valid?
Directionally yes for the broad SMAPE result. Not final-paper clean because LRPM is numerically unstable and the original fast/slow audit was weak.

2. Did any model calculation contain a bug?
Yes: `CP_REPO_LRPM` is rank-deficient and unregularized, producing numerically pathological predictions. Also the manual conditional summary has an `n_assets` metadata bug.

3. Is the manual summary correct?
Overall yes except floating point noise. Conditional performance columns yes, but `n_assets` is wrong on every row.

4. Does `CP_REPO_FRESH` correctly reproduce old CP?
Yes. All 10 assets pass; AAPL/SPY/QQQ spot checks align exactly by timestamp and target with mean absolute diffs around `1e-12`.

5. Does fast-vs-slow equivalence need stronger testing?
The artifact audit did. I added a stronger 18-model, 500-row, 3-asset audit and it passed.

6. Are feature definitions pure?
Yes. No CPB leakage, raw PM leakage, ungated leakage, CP omission, or placebo leakage found.

7. Are ridge penalties correctly applied?
Yes. Intercept and repo CP/RV controls are unpenalized; extra features are penalized for ridge models.

8. Is the gate meaningful or placebo-like?
The real gate is meaningful: it separates entry/recent-spike regimes from exit/older-spike regimes. But always-on gated linear models still lose overall; selective overlays work better.

9. Why does CP beat all challengers under SMAPE?
Because extra features create helpful corrections in rare regimes and harmful corrections elsewhere, with win rates below 42%. SMAPE's relative-error weighting makes the harmful rows dominate the always-on objective.

10. Is PM conditionally useful?
Yes, especially in entry/recent-spike/ramp regimes. It is not useful as an always-on broad model.

11. Is PM-specific structure supported or rejected?
Weakly supported at best. Placebos and generic shapes are close; residualized PM-specific signal is small.

12. Is generic path shape stronger than PM-specific residue structure?
Usually yes. Generic recent-slope/Haar/OPS-C/HG behavior is more defensible than a prime-specific residue claim.

13. What research claim is mathematically defensible?
"Repo CP is the best always-on SMAPE benchmark among this ladder; PM/path-shape features have conditional value in rare entry/recent-spike regimes, best used as a selective overlay."

14. What should be changed next?
Fix or ridge-regularize LRPM; keep the stronger fast/slow audit; fix manual conditional `n_assets`; widen/tune the lambda grid; report MAE/MSE/RMSE alongside SMAPE; develop a gate-threshold overlay rather than another always-on linear challenger.

## Final Conclusion

Results are directionally valid but audits need strengthening before final paper claims.
