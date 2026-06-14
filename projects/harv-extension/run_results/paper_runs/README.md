# Paper Runs

Each run folder preserves predictions, tables, figures, and LaTeX fragments generated during the paper workflow.

## Current And Supporting Runs

- `section5_cloud_push/` - Final Section 5 cloud run, including predictions, Section 5 tables, and PM/CP-CJ figures.
- `section5_robust_push/` - Robust Section 5 ablation summaries, figures, and status files used to build final robustness assets.
- `section5_cloud_seed/` - Seed/snapshot output for the Section 5 cloud workflow.
- `test_7/` - Multi-asset run used by `code/Helpers_for_paper_export_scripts/make_journal_figures.py` for the final journal figures outside Section 5.

## Historical Tests

- `test_1/` through `test_6/` - January 2026 exploratory and intermediate runs. These are preserved as historical outputs.
- `test_8_section5/` and `test_8_section5_AAPL_snapshot_20260506_163517/` - May 2026 Section 5 snapshots. These are preserved as historical outputs.

## Metadata

- `latest.json` currently points to `test_2`. Treat it as historical metadata from the interactive pipeline unless you intentionally regenerate or update the pipeline state.

Several helper scripts hard-code `test_7` and `section5_robust_push`; keep those folders in place unless the scripts are updated at the same time.
