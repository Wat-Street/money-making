# PM-Native Geometry Post-Validation

Source PM-native run: `28733449635`  
Post-validation GitHub Actions run: `28735116692`  
Branch: `cp-pm-incremental-actions`  
Assets: `AAPL,AMZN,EEM,FXI,GLD,GOOGL,HYG,QQQ,SPY,TLT`

## What Was Tested

This package validates the warmup-600 PM-native geometry run against corrected `CP_REPO_FRESH`, focusing on `PM_QDK_2`, `PHQO`, `PM_QDK`, and three validation-only matched placebos:

- `PM_QDK_2_RANDOM_RESIDUE_PLACEBO`
- `PM_QDK_2_SHUFFLED_LAG_PLACEBO`
- `PM_QDK_2_HAAR_SHAPE_PLACEBO`

The validation covers CP reproduction, no-lookahead, hyperparameter-selection timing, OOS alignment, CP feature leakage, robust loss subsets, Prop-3-style conditional subsets, error decomposition, paired tests, bootstrap intervals, and PM residual-geometry diagnostics.

## Result

Fatal audits passed. `PM_QDK_2` keeps its global SMAPE win over `CP_REPO_FRESH`, but it does **not** survive the non-placebo requirement.

- Pooled SMAPE advantage vs CP: `+1.47855`.
- Positive SMAPE assets: `9/10`.
- Pooled SMAPE bootstrap CI: `[1.39139, 1.56417]`.
- Paired one-sided p-value: `1.46443e-244`.
- Matched pooled SMAPE placebos beaten: `1/3`.

The placebo result is the limiting fact: random-residue and Haar/shape matched kernels slightly beat `PM_QDK_2` on pooled SMAPE, so the PM-specific geometry claim is not validated by this run.

## Review Status

This package is ready for repo2 internal review as a verified negative/non-placebo result. It should stay out of repo3 because it is diagnostic validation material, not draft-facing paper output.

Key files:

- `validation_summary.md`
- `metric_robustness.csv`
- `placebo_comparison.csv`
- `asset_failure_analysis.csv`
- `error_decomposition.csv`
- `statistical_tests.csv`
- `audit_summary.csv`
- `prop3_subset_performance.csv`
- `theory_diagnostics.csv`
- `original_stage4_report.md`
