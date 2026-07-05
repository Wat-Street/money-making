# PM-Native Geometry Post-Validation Summary

Fatal audits passed: `True`.
`PM_QDK_2` pooled SMAPE advantage vs `CP_REPO_FRESH`: `1.47855`.
`PM_QDK_2` positive SMAPE assets: `9/10`.
`PM_QDK_2` pooled SMAPE bootstrap CI: `[1.39139, 1.56417]`; paired one-sided p-value `1.46443e-244`.
`PM_QDK_2` beats matched SMAPE placebos pooled: `1/3`.

Conclusion: `PM_QDK_2` SMAPE win survives this validation: `False`.

Repo2 review status: ready only if fatal audits passed and validation package was copied by the promotion step.
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
