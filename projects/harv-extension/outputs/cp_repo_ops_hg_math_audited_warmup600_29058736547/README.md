# Prop 4 Mathematical-Fidelity Full Run

This is the cleaned repo-1 audit package for GitHub Actions run
[`29058736547`](https://github.com/Harman6139/money-making/actions/runs/29058736547)
at commit `03e80b3b70322f05196f31a13f079fe6762d87cb`.

## Configuration

- Assets: `AAPL,AMZN,EEM,FXI,GLD,GOOGL,HYG,QQQ,SPY,TLT`
- Models: all 18 Prop 4 models
- Benchmark: `CP_REPO_FRESH`
- `n=22`, `warmup=600`, `seed=42`
- Ten parallel asset jobs
- Repo-CP reproduction, no-lookahead, and fast-versus-direct audits enabled

This package has not been promoted to repo 2 or repo 3. The complete saved
prediction panels remain in the GitHub Actions artifact; this folder retains
the cleaned results, audit tables, paper tables, figures, registry, and proof.

## Independent Verification

- `180/180` model-asset fast/direct comparisons passed.
- Maximum fast/direct prediction difference: `7.88453504e-15`.
- `60/60` no-lookahead checks passed.
- `10/10` repo-CP reproduction checks passed; maximum absolute difference was
  `5.18248638e-16`.
- All 180 model-asset metadata records completed; no run failures occurred.
- All predictions were finite and every timestamp/actual alignment check passed.
- No `CPB_B*` diagnostic feature entered any of the 18 models.
- Direct row-level recalculation matched all 72 pooled SMAPE/MAE/MSE/RMSE rows.
- Direct pooled mean and median SMAPE calculations matched all 18 models.

## Corrected Full-Run Result

`CP_REPO_FRESH` remains best on pooled SMAPE at `100.879965`. The closest
challenger is `CP_REPO_OPS_K` at `100.965245`. `CP_REPO_OPS_HG` improves the
other three pooled losses (`MAE 0.000581548`, `MSE 2.177516e-06`,
`RMSE 0.001475641`) versus CP (`0.000586263`, `2.195810e-06`, `0.001481827`),
but its SMAPE is worse at `101.024762`. Thus no Prop 4 model dominates corrected
CP across all four losses.

The prior full-run outputs for unregularized OPS-R/LRPM/OPS-C and the old
random-gate placebo are superseded because they predate the stable QR solver
and causal random-gate repair.

See `results/model_construction_proof.md`,
`results/fast_slow_equivalence_audit.csv`,
`results/no_lookahead_audit.csv`, and
`results/alternative_loss_summary.csv` for the detailed evidence.
