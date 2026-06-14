# HAR-V Extension

This repository is organized around five top-level questions: where is the final paper, what generated it, where are the empirical results, where is the source data, and what configuration is needed to rerun it.

## Top-Level Map

| Folder | What belongs here | Use it when |
|---|---|---|
| `final_paper/` | Submission-ready manuscript, final PDF, and the exact table/figure assets used by that manuscript. | You want to read, compile, or upload the final paper. |
| `paper_outputs/` | Alternate manuscript export, working paper assets, previews, supplemental figures, and archived generated paper outputs. | You want generated paper material that is not the final submission bundle. |
| `run_results/` | Empirical predictions, tables, figures, LaTeX fragments, and historical test runs. | You want model outputs or evidence from a specific run. |
| `code/` | Python source code, paper export scripts, and shared utilities. | You want to rerun models or regenerate figures/tables. |
| `data/` | Raw and cleaned market data. | You need source input data for the empirical pipeline. |
| `config/` | Environment requirements and command notes. | You need setup or reproduction commands. |
| `references/` | Literature and reference PDFs. | You need background/reference material. |

## Final Paper

- Main manuscript: `final_paper/paper_quantfin_review.tex`
- Final compiled PDF: `final_paper/final_pdf/paper_quantfin_review.pdf`
- Final paper assets:
  - `final_paper/assets/figures_journal/`
  - `final_paper/assets/tables_journal/`

Build from `final_paper/`:

```powershell
latexmk -pdf -outdir=final_pdf paper_quantfin_review.tex
```

## Generated Paper Outputs

- `paper_outputs/alternate_paper.tex` - Alternate manuscript export.
- `paper_outputs/paper_assets/figures/` - Working figure exports and source-format companions.
- `paper_outputs/paper_assets/tables/` - Working/generated table exports and source CSV companions.
- `paper_outputs/paper_assets/figures/previews/` - Preview-only images.
- `paper_outputs/paper_assets/figures/supplemental_section5_ablation/` - Preserved supplemental Section 5 charts.
- `paper_outputs/archive/` - Daily probes and old build products kept for traceability.

## Empirical Run Results

- `run_results/current_intraday/` - Current generated intraday scratch/output area used by scripts.
- `run_results/paper_runs/section5_cloud_push/` - Final Section 5 cloud run outputs.
- `run_results/paper_runs/section5_robust_push/` - Robust Section 5 ablation outputs.
- `run_results/paper_runs/test_1/` through `test_8.../` - Historical tests preserved as historical evidence.

See `run_results/paper_runs/README.md` for run-level notes.

## Code And Config

- `code/run_interactive_pipeline.py` - Interactive pipeline.
- `code/intraday_benchmark.py` and `code/daily_benchmark.py` - Benchmark entry points.
- `code/utils/` - Shared helpers.
- `code/paper_export/` - Section-specific table/figure renderers.
- `code/Helpers_for_paper_export_scripts/` - Later-stage paper asset and robustness helpers.
- `config/requirements.txt` - Python dependencies.
- `config/result_generation_commands.md` - Command notes for regenerating results.

## Safety Notes

- Treat `data/market_data/` as source data.
- Treat `final_paper/assets/` as final manuscript inputs.
- Do not manually edit empirical values in generated tables or figures unless you are intentionally correcting the generation workflow.
- Do not move `run_results/paper_runs/test_7` or `run_results/paper_runs/section5_robust_push` without updating helper scripts that read those folders.
