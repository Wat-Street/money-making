# PM-Native Warmup-600 Results Report

Generated: 2026-07-05T07:43:22.982822Z

Benchmark: `CP_REPO_FRESH` using repo `RV*` and repo `CP_*` features from `contig_prime_modulo(vol.copy(), n=22, per_day_normalize=False)`.
PM-native kernels use only prior rows; the configured rolling support cap is recorded in `run_metadata.csv` as `pm_kernel_train_window`.

Positive advantage means lower loss than corrected CP.

## Global Loss Summary

```text
model_name loss_metric  n_obs_total  pooled_CP_loss  pooled_model_loss  pooled_advantage_CP_minus_model  assets_positive
      PHQO         MAE       356403        0.000596           0.000596                     4.910880e-08                7
  PM_QDK_2         MAE       367468        0.000586           0.000601                    -1.500762e-05                7
    PM_QDK         MAE       367468        0.000586           0.000616                    -3.015643e-05                1
      PHQO         MSE       356403        0.000002           0.000002                     5.883168e-11                7
    PM_QDK         MSE       367468        0.000002           0.000002                    -2.510489e-07                0
  PM_QDK_2         MSE       367468        0.000002           0.000003                    -3.555452e-07                0
      PHQO        RMSE       356403        0.001404           0.001404                     1.591853e-08                7
    PM_QDK        RMSE       367468        0.001389           0.001467                    -7.797657e-05                0
  PM_QDK_2        RMSE       367468        0.001389           0.001513                    -1.239225e-04                0
  PM_QDK_2       SMAPE       367468      100.879965          99.401412                     1.478553e+00                9
      PHQO       SMAPE       356403       97.970929          97.955374                     1.555521e-02                8
    PM_QDK       SMAPE       367468      100.879965         102.617596                    -1.737631e+00                0
```

## Asset Consistency

```text
model_name benchmark_model  n_assets  assets_positive_smape_advantage  assets_positive_mae_advantage  equal_weight_mean_smape_advantage  equal_weight_mean_abs_error_advantage
    PM_QDK   CP_REPO_FRESH        10                                0                              1                          -1.737737                          -3.015700e-05
  PM_QDK_2   CP_REPO_FRESH        10                                9                              7                           1.478078                          -1.502493e-05
      PHQO   CP_REPO_FRESH        10                                6                              7                          -0.000603                           4.743528e-08
```

