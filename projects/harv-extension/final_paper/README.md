# Final Paper

This folder is the submission-ready paper bundle.

## Contents

- `paper_quantfin_review.tex` - Main manuscript source.
- `final_pdf/paper_quantfin_review.pdf` - Latest compiled PDF.
- `assets/figures_journal/` - Exact figure PDFs included by the main manuscript.
- `assets/tables_journal/` - Exact LaTeX table fragments included by the main manuscript.

## Build

Run from this folder:

```powershell
latexmk -pdf -outdir=final_pdf paper_quantfin_review.tex
```

This folder is the one to use when preparing an Overleaf upload or final PDF check.
