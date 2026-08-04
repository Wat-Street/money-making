# PM-Native Geometry Discriminating Validation Summary

Fatal audits passed: `True`.
`PM_QDK_2` pooled SMAPE advantage vs `CP_REPO_FRESH`: `1.47855`.
`PM_QDK_2` positive SMAPE assets: `9/10`.
`PM_QDK_2` pooled SMAPE bootstrap CI: `[1.39139, 1.56417]`; paired one-sided p-value `1.46443e-244`.
`PM_QDK_2` beats matched SMAPE placebos pooled: `2/6`.

Conclusion: `PM_QDK_2` satisfies the full PM-specificity success bar: `False`.

Success-bar details are in `success_bar_summary.csv`.

Repo2 review status: do not promote unless the user explicitly approves this run after review.
Repo3 status: keep out of repo3; this is diagnostic/internal-review material, not draft-facing.

Required artifacts:
- `metric_robustness.csv`
- `placebo_comparison.csv`
- `asset_failure_analysis.csv`
- `error_decomposition.csv`
- `statistical_tests.csv`
- `audit_summary.csv`
- `prop3_subset_performance.csv`
- `theory_diagnostics.csv`
- `pm_ablation_comparison.csv`
- `success_bar_summary.csv`
