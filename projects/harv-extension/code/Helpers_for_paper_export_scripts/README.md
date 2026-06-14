# Paper Asset Helper Scripts

This folder keeps later-stage helpers used to assemble or post-process paper assets.

The folder name is intentionally left as `Helpers_for_paper_export_scripts` because several scripts call it by this path.

## Contents

- `make_journal_figures.py` - Builds the journal-ready figure PDFs from stored run outputs.
- `sync_paper_formats.py` - Synchronizes the alternate and QuantFin-style manuscript exports.
- `Section_2/` - Regime-analysis helper code.
- `Section_4/` - Error-advantage computation helper code.
- `Section_5/` - Robustness, ablation, capacity, and Section 5 table-building helpers.
- `Section_6/` - Asset-group significance helper code.

These scripts generally consume stored outputs from `run_results/paper_runs/` and write generated assets under `paper_outputs/paper_assets/` or `final_paper/assets/`, depending on whether the asset is working material or a final manuscript input.
