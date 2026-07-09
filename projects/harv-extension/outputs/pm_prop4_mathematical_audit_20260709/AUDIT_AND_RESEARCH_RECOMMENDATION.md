# Prop 4 Mathematical Audit And PM Diagnosis

## Outcome

- All `18` Prop 4 models pass finite-output, timestamp/target alignment, feature-registry, and fast-versus-direct solver checks after the QR and causal-gate repairs.
- Maximum fast/direct prediction difference in the smoke audit: `6.78992e-15`.
- `CP_REPO_FRESH` reproduction remains unchanged within `2.36356e-17` in the solver audit.
- Existing full-run results for unregularized OPS-R/LRPM/OPS-C and the old random-gate placebo are superseded and require a fresh full run.

## Why PM_QDK_2 Has The Observed Loss Pattern

On `367468` aligned rows, `PM_QDK_2` changes pooled SMAPE from `100.879965` to `99.401412`, but changes MAE from `0.000586262733` to `0.000601270358` and true pooled RMSE from `0.00148182658` to `0.0015972962`.

The model is SMAPE-native, and its Bayes action is lower than the conditional mean. Its mean forecast bias is `-9.51773875e-05` versus CP's `-3.5346397e-06`. The upper half of realized volatility explains `96.16%` of PM's excess squared error; the upper quartile explains `95.43%`. This is a location/action mismatch, not evidence that PM predicts tails well.

## Why Prime Specificity Is Not Established

1. Coordinate-wise quotient scaling cancels the heat factors to `3.48e-15` numerical error. The intended prime-diffusion smoothing is therefore absent from the actual distance.
2. True PM has standardized effective rank `10.845` in the controlled synthetic audit, while the six placebos range from `8.911` to `19.156`. Equal stored column count is not equal complexity.
3. Expanding quotient residuals use vintage-specific projection maps, so historical and query points are not represented in one common CP-fiber chart.
4. The CP distance is diagonal-scaled although `C` has rank `10/13` and the theory specifies a covariance-pseudoinverse metric.
5. The selected PM_QDK_2 bandwidth is at the broadest grid value for nearly every asset. Combined with the identical SMAPE action used by all geometries, most of the gain is generic smoothing/action regularization.
6. The previous CP-fiber success flag compared two negative correlations. The corrected criterion requires a positive PM slope before comparing it with placebos.

## Decisive PM Embedding Test

1. Freeze the PM operator, estimator, tau grid, rank, loss action, and selection rule before looking at the final period.
2. Represent every geometry by a `22 x d` operator and spectrally match its singular values, effective rank, trace, Frobenius norm, heat eigenvalue multiplicities, and kernel bandwidth budget to true PM.
3. Fit the train-only conditional projection once at each forecast origin and transform the query and every historical candidate with that same map.
4. Use the exact Mahalanobis CP metric and one scalar PM-Hilbert normalization, preserving relative heat weights.
5. Add a no-Q `(L,C)` kernel. PM must beat this ablation; otherwise the reported gain belongs to the common level/CP smoother and SMAPE action.
6. Use identical anchor-pair sets for every CP-fiber and smoothness comparison. Require positive fiber slope, lower PM residual Dirichlet energy, and low-mode held-out residual concentration.
7. Compare true PM with at least 1,000 spectral-matched random rotations plus the six structured placebos. Report PM's randomization percentile with family-wise correction.
8. Use purged chronological folds for tuning, then one untouched post-freeze time block. Infer with day/week moving-block bootstrap clustered by asset, plus the 10-asset sign/win test.

## Claim Threshold

Prime-specific value is supported only if PM beats the no-Q baseline and every structured placebo, ranks above the 95th percentile of spectral-matched rotations, has a positive CP-fiber slope and lower residual graph energy, wins at least 7/10 assets, and retains positive asset-clustered block-bootstrap intervals without a material MAE/RMSE loss.

The proposed practitioner model is specified separately in `PROPOSED_PM_CAST_D.md`.
