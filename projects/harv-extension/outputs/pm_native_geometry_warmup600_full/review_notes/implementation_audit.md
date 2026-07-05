# PM-Native Warmup-600 Implementation Audit

Source artifact: GitHub Actions run `28733449635`, copied from
`outputs/pm_native_geometry_warmup600_actions/28733449635_combined`.

Repo-2 promotion status: approved by user request on 2026-07-05.

## Scope

Models promoted:

- `PM_QDK`
- `PM_QDK_2`
- `PHQO`

Benchmark:

- `CP_REPO_FRESH = RV/HAR + contig_prime_modulo(vol.copy(), n=22, per_day_normalize=False)`
- Feature contract is repo `RV*` plus repo `CP_*`; no `CPB_B*`, `PM_*`, OPS, gate, or placebo columns in the benchmark.

Run settings:

- `n=22`
- `warmup=600`
- `seed=42`
- assets: `AAPL,AMZN,EEM,FXI,GLD,GOOGL,HYG,QQQ,SPY,TLT`

## Automated Audit Status

All aggregate audit checks passed:

- full asset coverage
- complete metadata rows
- empty run failures
- repo CP reproduction
- fast-vs-slow equivalence
- strong fast-vs-slow equivalence
- no-lookahead audit
- global level-sum audit

## Implementation Fidelity

The core PM-native operators match the requested implementation plan:

- `C` is a HAR plus prime-residue averaging path projection matrix.
- `M_C = I - C.T @ pinv(C @ C.T) @ C`.
- Prime-torus characters are built over `Z2 x Z3 x Z5`.
- Product-cycle eigenvalues use `sum_p(2 - 2cos(2*pi*a_p/p))`.
- `PM_QDK` uses the CP-quotient residual, diffusion-weighted nonzero harmonics, and a CP-state kernel deformed by PM harmonic inner products.
- `PM_QDK_2` uses log-centered paths, multiscale heat embeddings, train-only expanding residualization against `C`, and a SMAPE-native weighted action.
- `PHQO` uses the low-mode heat operator, maps quotient energy back to lag space, and applies Gibbs weighting.

## Caveats And Mistakes To Track

1. The PM kernel forecasts use only prior rows, but they cap the historical support at `pm_kernel_train_window=2500` for tractability. This is leakage-safe and recorded in `run_metadata.csv`, but it is not the literal all-prior kernel from the mathematical ideal.

2. `C` is the path-level HAR plus prime-residue projection used for quotient geometry. It is compatible with the repo CP benchmark, but it is not the exact fitted `CP_REPO_FRESH` regression operator. The quotient removes CP-style path averages, not the full learned CP forecast.

3. `PM_QDK_2` residualizes each row's PM embedding with an expanding train-only fit. This preserves no-lookahead, but `Q` values from different dates are generated under different historical projection fits. That is an online approximation to conditional quotient geometry, not a single global Hilbert-space projection.

4. `PHQO` was implemented with the plan's repo-compatible CP nesting: it scales the PHQO path ratio by the current `CP_REPO_FRESH` prediction so `gamma=0` returns exact CP. The PHQO attachment's raw final forecast is `<w(x_t), x_t>`, so this is a deliberate plan-approved benchmark-nesting modification.

5. PHQO has fewer aligned rows on some assets because rows with invalid raw base-path denominators or unavailable CP-scaled forecasts are skipped. This affects strict cross-model row comparability for PHQO, although the audit passes on the rows actually compared.

6. Placebo comparison output exists but is empty. The full PM-native workflow ran `CP_REPO_FRESH,PM_QDK,PM_QDK_2,PHQO` only, so the original Stage 4 placebo-comparison requirement was not satisfied in this run.

## Result Interpretation

No PM-native model beat corrected `CP_REPO_FRESH` globally on all four requested metrics.

- `PM_QDK_2` is the only substantive win: SMAPE advantage `+1.478553` over `367,468` rows and positive SMAPE on `9/10` assets. It loses globally on MAE, MSE, and RMSE.
- `PHQO` is effectively CP-nested. Its pooled gains are tiny and not evidence of a robust new signal.
- `PM_QDK` loses on all headline metrics.

Concise underperformance reason: the PM geometry mostly changes neighbor selection or path weighting inside a very strong CP level/persistence benchmark. That can improve the SMAPE action by shrinking relative-error behavior, but the residual PM shape signal is not stable enough to improve magnitude calibration and tail errors globally. In PHQO, validation often selects `gamma=0`, which means the data prefers the CP fallback over prime-harmonic deformation.
