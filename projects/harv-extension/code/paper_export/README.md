# Paper Export Scripts

This folder contains section-specific scripts that render publication tables and figures from generated result CSVs.

## Sections

- `Section_1/latex_table/` - Overall performance tables.
- `Section_2/` - Regime table and regime figure exports.
- `Section_3/figure/` - Temporal-stability figure exports.
- `Section_4/render_and_figure/` - Error-advantage summary and figure export.
- `Section_5/` - Parameter efficiency, capacity, random-control, and ablation figure/table exports.
- `Section_6/figure_and_latex_table/` - Asset-group figure and table export.

Most scripts are designed to be run from the project root. They read from `run_results/` and write figures, tables, or LaTeX fragments back to `run_results/` or `paper_outputs/`.
