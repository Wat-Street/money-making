# PM Selective Overlay Tests

The overlay uses saved prediction panels and applies challenger perturbations only when a gate is active.

Formula: `overlay = CP + active_flag * (challenger - CP)`.

Top-percentile gates use `abs(challenger - CP)` from saved forecast-time predictions. Validation-selected thresholds use the initial chronological prefix only and evaluate on later rows.

Condition strategies are ex-post descriptive and not tradable online labels unless rebuilt with train-only thresholds.

Main outputs live in `results/`; the compact sorted table is `paper_tables/selective_overlay_summary.csv`.
