# PM-CAST-D Verified Warmup-600 Run

GitHub Actions run:
[`29069772802`](https://github.com/Harman6139/money-making/actions/runs/29069772802)

- Commit: `a85471beff7f7b8eb84c54bc73bc2ae3617fd671`
- Assets: `AAPL,AMZN,EEM,FXI,GLD,GOOGL,HYG,QQQ,SPY,TLT`
- `n=22`, `warmup=600`, `seed=42`
- OOS observations: `367468`
- Selection origins: `222..621`; first final OOS origin: `622`
- Final OOS outcomes were not used to select or revise the model.

The full prediction panel is retained in the Actions artifact and is not copied
into git. This folder contains the frozen configuration, complete tuning grid,
cleaned results, audits, bootstrap inference, and figures.

## Frozen PM Configuration

```text
tau=0.5, kappa=0.12, gamma=-1.0
intercept=0.02, alpha=0.75, clip=0.1
tail_rule=none, low_modes=12
```

The PM transport was selected rather than the exact-CP fallback. Its pre-OOS
pooled normalized gains were positive for SMAPE, MAE, and MSE.

## Final Pooled Result

| Metric | CP_REPO_FRESH | PM_CAST_D | CP minus PM | Relative change |
|---|---:|---:|---:|---:|
| SMAPE | 100.879965 | 100.322612 | +0.557353 | +0.5525% |
| MAE | 0.000586263 | 0.000572067 | +0.000014196 | +2.4215% |
| MSE | 2.195810e-06 | 2.197910e-06 | -2.100346e-09 | -0.0957% |
| RMSE | 0.001481827 | 0.001482535 | -7.085322e-07 | -0.0478% |

PM-CAST-D therefore does **not** meet the all-fronts objective. It establishes
robust application value for SMAPE and MAE, but not for squared-error losses.

## Inference And Consistency

- SMAPE block-bootstrap 95% CI: `[+0.225289, +0.880518]`.
- MAE block-bootstrap 95% CI: `[+1.07878e-05, +1.74622e-05]`.
- MSE block-bootstrap 95% CI: `[-4.67214e-09, -8.75619e-11]`.
- RMSE block-bootstrap 95% CI: `[-1.37392e-06, -3.36666e-08]`.
- Asset wins: SMAPE `7/10`, MAE `10/10`, MSE `2/10`, RMSE `2/10`.

The PM model beats the calibration-only, spectral-random, and contiguous
controls on pooled SMAPE and MAE. It does not beat every control on MSE/RMSE.

## Failure Mechanism

Mean forecast bias is `-4.49533e-05` for PM-CAST-D versus `-3.53464e-06` for
CP. The upper half of realized volatility contributes `10.71x` the net excess
squared error; improvements in the lower half offset most, but not all, of that
tail loss. A second variant was not selected after viewing this OOS result.

## Distribution And Audits

- Zero-inflated 90% interval empirical coverage: `94.2596%` pooled.
- `230/230` asset/model mathematical and causality audits passed.
- All predictions and 5/50/95 percentiles are finite and ordered.
- Independent direct metric recomputation matches saved summaries within
  `1.42109e-14` maximum absolute difference.
- Exact CP nesting, CP-space annihilation, same-spectrum controls, strictly
  prior rank gates, and vector/scalar equivalence all pass.

This is a completed repo-1 result with `RUN_COMPLETE_NEEDS_REVIEW.txt`, not a
paper-success marker and not an approved repo-2 promotion.

