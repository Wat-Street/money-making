# PM/CP Research Internal Package

Generated: 2026-07-04T19:44:14Z

This package consolidates the current internal evidence for PM/path-order tests in the HAR/RV repo.
It is not a final paper draft. Prop 5 overlay work is included only as exploratory context for later review.

## Source Runs

- Prop 3 raw PM evidence: `outputs/cp_pm_incremental_tests/`.
- Prop 4 corrected repo-CP ladder: `outputs/cp_repo_ops_hg_actions/28269511971/`.
- Prop 5 selective overlay evidence: `outputs/cp_repo_ops_hg_actions/28269511971/audit_recomputed/switching_overlay_summary.csv`.
- Reproducible overlay test suite: `outputs/pm_selective_overlay_tests/`.

The package deliberately does not copy large full prediction panels. `source_runs/source_run_manifest.csv` records the source paths.

## Main Takeaways

- Prop 1 and Prop 2 are theoretical support statements.
- Prop 3 is empirically supported: raw/global PM is noisy overall, while entry and recent-spike regimes can be positive.
- Prop 4 is not proven globally: always-on PM/shape challengers do not beat repo CP under headline SMAPE.
- Prop 5 is a promising exploratory lead: selective overlays can produce positive equal-weight SMAPE advantage on rare active rows and across most assets, but threshold validation needs manual review before draft use.
- `CP_REPO_LRPM` is diagnostic/non-paper-eligible in the current corrected run because the unregularized design is rank deficient and numerically unstable.

## Key Files

- `prop3_raw_pm_diagnostics/`: cleaned Prop 3 diagnostics.
- `prop4_corrected_repo_cp_ladder/`: corrected repo-CP 18-model evidence.
- `prop5_overlay_exploratory/`: exploratory overlay context only.
- `model_guide.md`: compact model-family guide for teammates.
- `test_results_guide.md`: what was tested, what passed, and what remains weak.
- `prop_status_summary.md`: proposition-by-proposition status.
- `tables/prop4_overall_smape_summary.csv`: corrected 18-model always-on ladder.
- `tables/prop5_selective_overlay_summary.csv`: strongest overlay evidence.
- `tables/prop5_selective_overlay_recomputed_from_predictions.csv`: regenerated overlay suite from saved prediction panels.
- `audits/`: fast/slow, feature-purity, ridge, and CP reproduction checks.
- `latex/internal_results_summary.tex`: minimal internal write-up skeleton.

## Interpretation Guardrails

- If selective gated overlays beat CP, temporal path shape matters after CP.
- If CP-OPS-HG beats gated OPS-C, PM residue structure adds value beyond local shape.
- If OPS-R fails against random/shuffled residues, PM-specific residue structure is not supported.
- If Haar or recent slope matches OPS-C, the result is generic path shape rather than PM-specific.
- Positive mean advantage with sub-50% win rate is a gain-size effect, not dominance.
- If CP-OPS-HG loses to gated OPS-C, OPS-R is contaminating the final hybrid and should be reported as rejected.
