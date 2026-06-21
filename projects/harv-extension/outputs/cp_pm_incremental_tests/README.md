# CP + PM Incremental Value Test

Generated: 2026-06-21T03:08:02.148439Z

## Purpose

This run tests whether adding Prime-Modulo (PM) temporal-granularity features improves a Contiguous-Prime (CP) local-volatility baseline.

## Methodological Choice

Saved CP validation passed for 10/10 assets.

Because saved `Predicted_CP` could not be treated as perfectly reproducible for every asset, the primary interpretable test uses a **fresh apples-to-apples baseline**:

- Fresh CP: recomputed from raw data with current code/settings.
- Fresh CP+PM raw: recomputed from the same data with the same target, same warmup, same row alignment, and same expanding-window logic.

Mode used: `fresh_cp_vs_fresh_cp_plus_pm_raw`

This answers: holding the current pipeline fixed, does adding PM improve CP?

## Speed/fit implementation

The prior script timed out because it called statsmodels OLS at every timestamp. This run uses incremental normal-equation updates with the same expanding-window alignment. A tiny numerical ridge of `1e-12` is used only for stable solves. CP and CP+PM use identical solver logic.

## Anti-Lookahead Rule

All conditional states are built using lagged `Actual.shift(k)` values only, with k >= 1.

## Tickers Included

AAPL, AMZN, EEM, FXI, GLD, GOOGL, HYG, QQQ, SPY, TLT

## Validation Summary

ticker status                                   message  n_existing  n_recomputed  n_aligned  mean_existing_CP  mean_recomputed_CP  std_existing_CP  std_recomputed_CP  mean_abs_diff  median_abs_diff  max_abs_diff  correlation  first_aligned_date   last_aligned_date  passes
  AAPL passed diagnostic exact-saved-CP validation only       36739         36739      36739          0.000943            0.000943         0.000544           0.000544   7.551853e-13     1.242504e-13  2.519782e-10          1.0 2023-08-17T16:00:00 2025-07-07T16:00:00    True
  AMZN passed diagnostic exact-saved-CP validation only       36739         36739      36739          0.001109            0.001109         0.000557           0.000557   3.836122e-13     8.504081e-14  9.398805e-11          1.0 2023-08-17T16:00:00 2025-07-07T16:00:00    True
   EEM passed diagnostic exact-saved-CP validation only       36739         36739      36739          0.000421            0.000421         0.000170           0.000170   4.917242e-13     1.499647e-13  3.777881e-11          1.0 2023-08-17T16:00:00 2025-07-07T16:00:00    True
   FXI passed diagnostic exact-saved-CP validation only       36817         36817      36817          0.000791            0.000791         0.000336           0.000336   3.079965e-13     8.073815e-14  1.026217e-10          1.0 2023-08-17T16:00:00 2025-07-08T16:00:00    True
   GLD passed diagnostic exact-saved-CP validation only       36739         36739      36739          0.000537            0.000537         0.000210           0.000210   2.355574e-12     1.182422e-13  1.113829e-09          1.0 2023-08-17T16:00:00 2025-07-07T16:00:00    True
 GOOGL passed diagnostic exact-saved-CP validation only       36739         36739      36739          0.001046            0.001046         0.000459           0.000459   3.463277e-13     6.871457e-14  1.106375e-10          1.0 2023-08-17T16:00:00 2025-07-07T16:00:00    True
   HYG passed diagnostic exact-saved-CP validation only       36739         36739      36739          0.000161            0.000161         0.000128           0.000128   2.021064e-12     2.977890e-13  4.286814e-10          1.0 2023-08-17T16:00:00 2025-07-07T16:00:00    True
   QQQ passed diagnostic exact-saved-CP validation only       36739         36739      36739          0.000663            0.000663         0.000392           0.000392   1.074580e-12     3.224109e-13  2.300085e-10          1.0 2023-08-17T16:00:00 2025-07-07T16:00:00    True
   SPY passed diagnostic exact-saved-CP validation only       36739         36739      36739          0.000487            0.000487         0.000363           0.000363   1.650994e-12     5.004427e-13  2.593738e-10          1.0 2023-08-17T16:00:00 2025-07-07T16:00:00    True
   TLT passed diagnostic exact-saved-CP validation only       36739         36739      36739          0.000580            0.000580         0.000176           0.000176   3.961816e-13     1.533403e-13  6.895342e-11          1.0 2023-08-17T16:00:00 2025-07-07T16:00:00    True

## Shift Diagnostic Preview

ticker  shift  n_aligned  mean_abs_diff  median_abs_diff  max_abs_diff  correlation  first_aligned_date   last_aligned_date
  AAPL     -2      36739   1.519814e-04     7.808365e-05  1.378095e-02     0.813130 2023-08-17T16:00:00 2025-07-07T16:00:00
  AAPL     -1      36739   1.232745e-04     6.068917e-05  1.413305e-02     0.847464 2023-08-17T16:00:00 2025-07-07T16:00:00
  AAPL      0      36739   7.551853e-13     1.242504e-13  2.519782e-10     1.000000 2023-08-17T16:00:00 2025-07-07T16:00:00
  AAPL      1      36739   1.232745e-04     6.068918e-05  1.413305e-02     0.847464 2023-08-17T16:00:00 2025-07-07T16:00:00
  AAPL      2      36739   1.519814e-04     7.808365e-05  1.378095e-02     0.813130 2023-08-17T16:00:00 2025-07-07T16:00:00
  AMZN     -2      36739   1.836218e-04     9.918628e-05  8.949716e-03     0.792058 2023-08-17T16:00:00 2025-07-07T16:00:00
  AMZN     -1      36739   1.448150e-04     7.584056e-05  8.807891e-03     0.847524 2023-08-17T16:00:00 2025-07-07T16:00:00
  AMZN      0      36739   3.836122e-13     8.504081e-14  9.398805e-11     1.000000 2023-08-17T16:00:00 2025-07-07T16:00:00
  AMZN      1      36739   1.448150e-04     7.584056e-05  8.807891e-03     0.847524 2023-08-17T16:00:00 2025-07-07T16:00:00
  AMZN      2      36739   1.836218e-04     9.918628e-05  8.949716e-03     0.792058 2023-08-17T16:00:00 2025-07-07T16:00:00
   EEM     -2      36739   4.890917e-05     2.171443e-05  3.236948e-03     0.751262 2023-08-17T16:00:00 2025-07-07T16:00:00
   EEM     -1      36739   3.610179e-05     1.557076e-05  3.328281e-03     0.844658 2023-08-17T16:00:00 2025-07-07T16:00:00
   EEM      0      36739   4.917242e-13     1.499647e-13  3.777881e-11     1.000000 2023-08-17T16:00:00 2025-07-07T16:00:00
   EEM      1      36739   3.610179e-05     1.557076e-05  3.328281e-03     0.844658 2023-08-17T16:00:00 2025-07-07T16:00:00
   EEM      2      36739   4.890917e-05     2.171443e-05  3.236948e-03     0.751262 2023-08-17T16:00:00 2025-07-07T16:00:00
   FXI     -2      36817   9.331918e-05     3.631776e-05  5.596667e-03     0.784821 2023-08-17T16:00:00 2025-07-08T16:00:00
   FXI     -1      36817   6.238031e-05     2.428092e-05  5.506442e-03     0.888861 2023-08-17T16:00:00 2025-07-08T16:00:00
   FXI      0      36817   3.079965e-13     8.073815e-14  1.026217e-10     1.000000 2023-08-17T16:00:00 2025-07-08T16:00:00
   FXI      1      36817   6.238031e-05     2.428092e-05  5.506442e-03     0.888861 2023-08-17T16:00:00 2025-07-08T16:00:00
   FXI      2      36817   9.331918e-05     3.631776e-05  5.596667e-03     0.784821 2023-08-17T16:00:00 2025-07-08T16:00:00

## Fresh Conditional Hybrid Summary

Positive advantage means fresh CP+PM raw beats fresh CP.

                   condition  total_n_obs  number_of_assets  equal_weight_asset_mean_advantage  equal_weight_asset_median_advantage  equal_weight_asset_hybrid_win_rate  pooled_CP_mean_SMAPE  pooled_HYBRID_mean_SMAPE  pooled_mean_Hybrid_advantage_vs_CP  pooled_hybrid_win_rate_vs_CP  number_of_assets_where_hybrid_beats_CP  asset_win_rate
            all_observations       367248                10                          -0.215225                             0.000000                            0.407794            100.863661                101.079029                           -0.215368                      0.407790                                       2             0.2
         cluster_entry_loose        73457                10                           0.246355                             0.014404                            0.449455             93.560889                 93.314574                            0.246315                      0.449447                                       9             0.9
        cluster_entry_strict        73266                10                           0.244476                             0.014205                            0.449644             93.513519                 93.269160                            0.244359                      0.449608                                       9             0.9
          cluster_exit_loose        73455                10                          -0.797490                            -0.143215                            0.385085             97.782703                 98.580508                           -0.797804                      0.385079                                       0             0.0
         cluster_exit_strict        73334                10                          -0.801490                            -0.144429                            0.385224             97.711896                 98.513671                           -0.801776                      0.385183                                       0             0.0
           inflection_points       144932                10                          -0.289302                            -0.015161                            0.424945             91.125650                 91.417986                           -0.292336                      0.427801                                       0             0.0
high_within_block_dispersion        68970                10                           0.060897                             0.000499                            0.378531            119.604272                119.562804                            0.041469                      0.388357                                       6             0.6
       recent_spike_position       124971                10                           0.238083                             0.069051                            0.423414            111.528013                111.303767                            0.224246                      0.403518                                       8             0.8
        older_spike_position       109235                10                          -0.446694                            -0.009616                            0.399700             97.520544                 97.976027                           -0.455483                      0.402490                                       0             0.0
 same_average_different_path         7396                10                          -0.118888                            -0.061960                            0.379950            117.373182                117.439007                           -0.065825                      0.364657                                       3             0.3

## Runtime Metadata Preview

ticker  raw_rows  vol_rows  cp_features  hybrid_features                                                                                                       cp_feature_names                                                                                                                                                                                                       hybrid_feature_names  elapsed_seconds status
  AAPL     38999     37362           13               23 [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, RV_d, RV_m, RV_w] [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, PM_k2_r0, PM_k2_r1, PM_k3_r0, PM_k3_r1, PM_k3_r2, PM_k5_r0, PM_k5_r1, PM_k5_r2, PM_k5_r3, PM_k5_r4, RV_d, RV_m, RV_w]         2.677000     ok
  AMZN     38999     37362           13               23 [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, RV_d, RV_m, RV_w] [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, PM_k2_r0, PM_k2_r1, PM_k3_r0, PM_k3_r1, PM_k3_r2, PM_k5_r0, PM_k5_r1, PM_k5_r2, PM_k5_r3, PM_k5_r4, RV_d, RV_m, RV_w]         2.645760     ok
   EEM     38999     37362           13               23 [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, RV_d, RV_m, RV_w] [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, PM_k2_r0, PM_k2_r1, PM_k3_r0, PM_k3_r1, PM_k3_r2, PM_k5_r0, PM_k5_r1, PM_k5_r2, PM_k5_r3, PM_k5_r4, RV_d, RV_m, RV_w]         2.657724     ok
   FXI     39077     37440           13               23 [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, RV_d, RV_m, RV_w] [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, PM_k2_r0, PM_k2_r1, PM_k3_r0, PM_k3_r1, PM_k3_r2, PM_k5_r0, PM_k5_r1, PM_k5_r2, PM_k5_r3, PM_k5_r4, RV_d, RV_m, RV_w]         2.650971     ok
   GLD     38999     37362           13               23 [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, RV_d, RV_m, RV_w] [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, PM_k2_r0, PM_k2_r1, PM_k3_r0, PM_k3_r1, PM_k3_r2, PM_k5_r0, PM_k5_r1, PM_k5_r2, PM_k5_r3, PM_k5_r4, RV_d, RV_m, RV_w]         2.641016     ok
 GOOGL     38999     37362           13               23 [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, RV_d, RV_m, RV_w] [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, PM_k2_r0, PM_k2_r1, PM_k3_r0, PM_k3_r1, PM_k3_r2, PM_k5_r0, PM_k5_r1, PM_k5_r2, PM_k5_r3, PM_k5_r4, RV_d, RV_m, RV_w]         2.625552     ok
   HYG     38999     37362           13               23 [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, RV_d, RV_m, RV_w] [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, PM_k2_r0, PM_k2_r1, PM_k3_r0, PM_k3_r1, PM_k3_r2, PM_k5_r0, PM_k5_r1, PM_k5_r2, PM_k5_r3, PM_k5_r4, RV_d, RV_m, RV_w]         2.626155     ok
   QQQ     38999     37362           13               23 [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, RV_d, RV_m, RV_w] [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, PM_k2_r0, PM_k2_r1, PM_k3_r0, PM_k3_r1, PM_k3_r2, PM_k5_r0, PM_k5_r1, PM_k5_r2, PM_k5_r3, PM_k5_r4, RV_d, RV_m, RV_w]         2.630660     ok
   SPY     38999     37362           13               23 [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, RV_d, RV_m, RV_w] [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, PM_k2_r0, PM_k2_r1, PM_k3_r0, PM_k3_r1, PM_k3_r2, PM_k5_r0, PM_k5_r1, PM_k5_r2, PM_k5_r3, PM_k5_r4, RV_d, RV_m, RV_w]         2.612737     ok
   TLT     38999     37362           13               23 [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, RV_d, RV_m, RV_w] [CP_k2_a0, CP_k2_a1, CP_k3_a0, CP_k3_a1, CP_k3_a2, CP_k5_a0, CP_k5_a1, CP_k5_a2, CP_k5_a3, CP_k5_a4, PM_k2_r0, PM_k2_r1, PM_k3_r0, PM_k3_r1, PM_k3_r2, PM_k5_r0, PM_k5_r1, PM_k5_r2, PM_k5_r3, PM_k5_r4, RV_d, RV_m, RV_w]         2.637801     ok

## Key Files

- `results/cp_recompute_validation.csv`
- `results/cp_alignment_shift_diagnostics.csv`
- `results/fresh_run_metadata.csv`
- `results/fresh_run_failures.csv`
- `results/fresh_cp_vs_hybrid_predictions_summary.csv`
- `results/fresh_conditional_hybrid_summary_by_asset.csv`
- `results/fresh_conditional_hybrid_summary_pooled.csv`
- `results/top_fresh_hybrid_conditions.csv`
- `figures/fresh_hybrid_advantage_by_condition.png`
- `figures/fresh_hybrid_win_rate_by_condition.png`
- `latex/fresh_conditional_hybrid_summary_pooled.tex`
