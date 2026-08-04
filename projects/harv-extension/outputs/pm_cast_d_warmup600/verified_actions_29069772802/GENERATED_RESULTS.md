# PM-CAST-D Warmup-600 Practitioner Results

Generated: 2026-07-10T04:52:59.752408+00:00

The configuration was frozen using origins strictly before the first warmup-600 OOS forecast.
Positive advantage means lower loss than `CP_REPO_FRESH`.

## Pooled Metrics

```text
             model_name loss_metric    CP_loss  model_loss  advantage_CP_minus_model  relative_advantage
          CP_REPO_FRESH       SMAPE 100.879965  100.879965              0.000000e+00            0.000000
          CP_REPO_FRESH         MAE   0.000586    0.000586              0.000000e+00            0.000000
          CP_REPO_FRESH         MSE   0.000002    0.000002              0.000000e+00            0.000000
          CP_REPO_FRESH        RMSE   0.001482    0.001482              0.000000e+00            0.000000
              PM_CAST_D       SMAPE 100.879965  100.322612              5.573529e-01            0.005525
              PM_CAST_D         MAE   0.000586    0.000572              1.419555e-05            0.024214
              PM_CAST_D         MSE   0.000002    0.000002             -2.100346e-09           -0.000957
              PM_CAST_D        RMSE   0.001482    0.001483             -7.085322e-07           -0.000478
 SPECTRAL_RANDOM_CAST_D       SMAPE 100.879965  100.702332              1.776323e-01            0.001761
 SPECTRAL_RANDOM_CAST_D         MAE   0.000586    0.000583              3.551528e-06            0.006058
 SPECTRAL_RANDOM_CAST_D         MSE   0.000002    0.000002             -1.574299e-09           -0.000717
 SPECTRAL_RANDOM_CAST_D        RMSE   0.001482    0.001482             -5.311070e-07           -0.000358
      CONTIGUOUS_CAST_D       SMAPE 100.879965  100.775041              1.049238e-01            0.001040
      CONTIGUOUS_CAST_D         MAE   0.000586    0.000583              2.915350e-06            0.004973
      CONTIGUOUS_CAST_D         MSE   0.000002    0.000002             -2.819246e-09           -0.001284
      CONTIGUOUS_CAST_D        RMSE   0.001482    0.001483             -9.509687e-07           -0.000642
CAST_D_CALIBRATION_ONLY       SMAPE 100.879965  100.680558              1.994066e-01            0.001977
CAST_D_CALIBRATION_ONLY         MAE   0.000586    0.000582              4.683301e-06            0.007988
CAST_D_CALIBRATION_ONLY         MSE   0.000002    0.000002             -3.742739e-10           -0.000170
CAST_D_CALIBRATION_ONLY        RMSE   0.001482    0.001482             -1.262826e-07           -0.000085
```

## Asset Consistency

```text
model_name loss_metric  assets_positive  assets_zero  assets_negative  n_assets
 PM_CAST_D         MAE               10            0                0        10
 PM_CAST_D         MSE                2            0                8        10
 PM_CAST_D        RMSE                2            0                8        10
 PM_CAST_D       SMAPE                7            0                3        10
```

## Moving-Block Inference

```text
model_name loss_metric  bootstrap_reps  block_days        ci_low       ci_high  bootstrap_mean_advantage  one_sided_p_advantage_le_zero
 PM_CAST_D       SMAPE            2000           5  2.252891e-01  8.805175e-01              5.576016e-01                       0.001000
 PM_CAST_D         MAE            2000           5  1.078776e-05  1.746216e-05              1.418969e-05                       0.000500
 PM_CAST_D         MSE            2000           5 -4.672143e-09 -8.756189e-11             -2.062556e-09                       0.982009
 PM_CAST_D        RMSE            2000           5 -1.373919e-06 -3.366657e-08             -6.807167e-07                       0.982009
```

## Distribution Calibration

```text
 asset  n_obs  nominal_coverage  empirical_coverage  mean_interval_width  median_interval_width  median_absolute_error
  AAPL  36739               0.9            0.935436             0.002546               0.002218               0.000726
  AMZN  36739               0.9            0.951795             0.003450               0.003075               0.000844
   EEM  36739               0.9            0.965377             0.001839               0.001709               0.000425
   FXI  36817               0.9            0.965016             0.002848               0.002604               0.000634
   GLD  36739               0.9            0.972563             0.002376               0.002244               0.000449
 GOOGL  36739               0.9            0.956776             0.003601               0.003293               0.000828
   HYG  36739               0.9            0.947005             0.000478               0.000368               0.000148
   QQQ  36739               0.9            0.883884             0.001770               0.001537               0.000482
   SPY  36739               0.9            0.867471             0.001476               0.001233               0.000358
   TLT  36739               0.9            0.980593             0.002456               0.002342               0.000436
POOLED 367468               0.9            0.942596             0.002284               0.002077               0.000533
```

## Success Bar

```text
                             check  passed                                                                                                                         detail
                  all_asset_audits    True                                                                                                                        230/230
        pooled_all_metrics_beat_cp   False                                                                                                                      SMAPE,MAE
     at_least_7_assets_each_metric   False                                                                                   {"SMAPE": 7, "MAE": 10, "MSE": 2, "RMSE": 2}
 bootstrap_ci_positive_all_metrics   False {"SMAPE": 0.22528907737684423, "MAE": 1.0787756736240463e-05, "MSE": -4.6721429978614996e-09, "RMSE": -1.3739191443635602e-06}
beats_calibration_only_all_metrics   False                                                                                                                      SMAPE,MAE
             pm_transport_selected    True                                                                                                                     alpha=0.75
```

Application value is established only by the observed OOS and audit checks; prime specificity still requires PM to beat the matched operators.
