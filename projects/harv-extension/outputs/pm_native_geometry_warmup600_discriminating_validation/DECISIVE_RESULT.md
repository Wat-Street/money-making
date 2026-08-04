# Decisive Validation Result

`PM_QDK_2` remains a real SMAPE challenger to corrected `CP_REPO_FRESH`, but this round does not prove prime-modular specificity.

Key results:

- Pooled SMAPE advantage vs `CP_REPO_FRESH`: `+1.4785526053608025`.
- Positive SMAPE assets: `9/10`; `GLD` is the miss.
- Pooled SMAPE bootstrap CI: `[1.39139289785069, 1.5641745047929214]`.
- Matched SMAPE placebo wins: `2/6`.
- Partial PM ablation wins: full PM beats `0/6`.
- CP-fiber test favors PM: `True`.
- Residual smoothness test favors PM: `False`.
- Fatal audits: all passed, including fast-vs-slow kernel equivalence with `max_abs_pred_diff=0.0`.

Concise reason for underperformance:

The SMAPE gain is not uniquely tied to full prime-modular CRT geometry. Several equal-complexity non-PM geometries and every partial-modulus ablation match or beat full `PM_QDK_2` on pooled SMAPE, and the true PM residual geometry does not make CP residuals smoother than the best placebo. That pattern says the benefit is more likely coming from generic path smoothing / geometry regularization, not from distinct full prime-modular structure beyond CP.
