# CP-REPO-OPS-HG Tests

Generated: 2026-07-10T00:08:23.286341Z

## Run

- Mode: `tables-only`
- Phase: `all`
- Assets: `AAPL,AMZN,EEM,FXI,GLD,GOOGL,HYG,QQQ,SPY,TLT`
- Models requested: `CP_REPO_FRESH,HAR_RV,RAW_PM,RAW_CP_REPO_PLUS_PM,CP_REPO_OPS_R,CP_REPO_RIDGE_OPS_R,CP_REPO_LRPM,CP_REPO_RECENT_SLOPE,CP_REPO_HAAR_SHAPE,CP_REPO_OPS_C,CP_REPO_RIDGE_OPS_C,CP_REPO_GATED_RIDGE_OPS_C,CP_REPO_OPS_HG,RANDOM_RESIDUES_PLACEBO_REPO,SHUFFLED_LAG_PM_PLACEBO_REPO,RANDOM_GATE_PLACEBO_REPO,RIDGE_AR22,CP_REPO_OPS_K`
- Seed: `42`
- Reuse prior baselines: `False`
- Skip existing: `False`
- Force rerun: `False`
- Target transform: `level`
- Ridge lambda grid: `0.001`
- Ridge R-ratio grid: `10`
- PM tau grid: `0.05,0.1,0.25,0.5,1.0`
- PM eta grid: `0,0.25,0.5,1.0`
- PM bandwidth grid: `0.5,1,2`
- PHQO gamma grid: `0,0.25,0.5,1,2`
- PM low modes: `12`
- PM kernel train window: `2500` prior rows (`0` means all prior rows)
- CV mode: `auto`
- Command: `outputs/cp_repo_ops_hg_tests/scripts/run_cp_repo_ops_hg_tests.py --mode tables-only --phase all --assets AAPL,AMZN,EEM,FXI,GLD,GOOGL,HYG,QQQ,SPY,TLT --models HAR_RV,CP_REPO_FRESH,RAW_PM,RAW_CP_REPO_PLUS_PM,CP_REPO_OPS_R,CP_REPO_RIDGE_OPS_R,CP_REPO_LRPM,CP_REPO_RECENT_SLOPE,CP_REPO_HAAR_SHAPE,CP_REPO_OPS_C,CP_REPO_RIDGE_OPS_C,CP_REPO_GATED_RIDGE_OPS_C,CP_REPO_OPS_HG,RANDOM_RESIDUES_PLACEBO_REPO,SHUFFLED_LAG_PM_PLACEBO_REPO,RANDOM_GATE_PLACEBO_REPO,RIDGE_AR22,CP_REPO_OPS_K --n 22 --warmup 600 --seed 42 --outdir outputs/cp_repo_ops_hg_math_audited_actions/29058736547/combined`
- Paper eligibility rule: rows are paper eligible only when both the model and CP benchmark were generated in the current full run and exact timestamp/actual alignment passes. Smoke, reused-prior, skipped, and scaffolded rows are excluded.

## Correct Repo CP Benchmark

`CP_REPO_FRESH` is the established repository CP benchmark: `contig_prime_modulo(vol.copy(), n=22, per_day_normalize=False)` with `RV*` columns plus repo `CP_*` columns, matching the original CP+PM incremental test while explicitly excluding diagnostic `CPB_B*` columns. The four-zone lag blocks `[[1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21, 22]]` are retained only as local shape/diagnostic scaffolds, not as the benchmark and not as proof of orthogonality to repo CP.

## Commands

Smoke test:

```powershell
python outputs/cp_repo_ops_hg_tests/scripts/run_cp_repo_ops_hg_tests.py --mode smoke --assets AAPL --phase all --seed 42 --force --audit-no-lookahead --audit-repo-cp-reproduction --audit-fast-slow-equivalence --cv-mode inner --lambda-grid 0.0001,0.001,0.01 --lambda-r-ratio-grid 2,5,10,20
```

Phase 1 run:

```powershell
python outputs/cp_repo_ops_hg_tests/scripts/run_cp_repo_ops_hg_tests.py --mode full --phase phase1 --seed 42 --skip-existing --cv-mode inner --lambda-grid 0.0001,0.001,0.01
```

Phase 2 / full linear ladder:

```powershell
python outputs/cp_repo_ops_hg_tests/scripts/run_cp_repo_ops_hg_tests.py --mode full --phase linear --seed 42 --skip-existing --cv-mode inner --lambda-grid 0.0001,0.001,0.01 --lambda-r-ratio-grid 2,5,10,20
```

Final all-model full run:

```powershell
python outputs/cp_repo_ops_hg_tests/scripts/run_cp_repo_ops_hg_tests.py --mode full --phase all --seed 42 --skip-existing --audit-no-lookahead --audit-repo-cp-reproduction --audit-fast-slow-equivalence --cv-mode inner --lambda-grid 0.0001,0.001,0.01 --lambda-r-ratio-grid 2,5,10,20
```

Regenerate paper tables from existing outputs:

```powershell
python outputs/cp_repo_ops_hg_tests/scripts/run_cp_repo_ops_hg_tests.py --mode tables-only --outdir outputs/cp_repo_ops_hg_tests
```

## Interpretation Guardrails

1. If `RAW_CP_REPO_PLUS_PM` loses overall but helps cluster-entry/recent-spike, raw PM is conditionally useful but globally noisy.
2. If `CP_REPO_GATED_RIDGE_OPS_C` beats `CP_REPO_FRESH` overall and improves entry/recent-spike without large exit/high-dispersion losses, cleaned path-shape is supported.
3. If `CP_REPO_OPS_HG` beats `CP_REPO_GATED_RIDGE_OPS_C`, then PM residue structure adds value beyond generic/local path shape.
4. If `CP_REPO_OPS_HG` loses to `CP_REPO_GATED_RIDGE_OPS_C`, then OPS-R/PM residue structure contaminates the final hybrid and should be rejected.
5. If `CP_REPO_RECENT_SLOPE` or `CP_REPO_HAAR_SHAPE` matches/beats OPS-C, then the signal is generic path shape rather than PM-specific.
6. If random residues or shuffled PM match true OPS-R, PM-specific residue structure is not supported.
7. If random gate matches real gate, the regime-targeting story is weak.
8. If results are positive mean advantage but win rate below 50%, describe them as gain-size effects, not row-wise dominance.
9. Do not claim PM is validated unless PM-specific models beat generic shape models and placebos in the expected regimes.
10. Do not use the four-block diagnostic benchmark for primary research claims.

## Shape Level-Removal Check

Shape global level-removal failures: `0`

## Ablation Preview

```text
                  model_name        condition  n_obs  mean_advantage_vs_CP  win_rate_vs_CP  is_paper_eligible
                      HAR_RV all_observations 367248             -1.287757        0.381309               True
               CP_REPO_FRESH all_observations 367248              0.000000        0.000000               True
                      RAW_PM all_observations 367248             -1.023949        0.401516               True
         RAW_CP_REPO_PLUS_PM all_observations 367248             -0.215369        0.407512               True
               CP_REPO_OPS_R all_observations 367248             -0.128398        0.412138               True
         CP_REPO_RIDGE_OPS_R all_observations 367248             -0.128306        0.411964               True
                CP_REPO_LRPM all_observations 367248             -0.200541        0.410096              False
        CP_REPO_RECENT_SLOPE all_observations 367248             -0.209797        0.413470               True
          CP_REPO_HAAR_SHAPE all_observations 367248             -0.254123        0.410701               True
               CP_REPO_OPS_C all_observations 367248             -0.258753        0.410665               True
         CP_REPO_RIDGE_OPS_C all_observations 367248             -0.258423        0.411172               True
   CP_REPO_GATED_RIDGE_OPS_C all_observations 367248             -0.154198        0.418110               True
              CP_REPO_OPS_HG all_observations 367248             -0.136635        0.412435               True
RANDOM_RESIDUES_PLACEBO_REPO all_observations 367248             -0.146958        0.417938               True
SHUFFLED_LAG_PM_PLACEBO_REPO all_observations 367248             -0.192249        0.414943               True
    RANDOM_GATE_PLACEBO_REPO all_observations 367248             -0.188410        0.409138               True
                  RIDGE_AR22 all_observations 367248             -0.498995        0.409399               True
               CP_REPO_OPS_K all_observations 367248             -0.083098        0.419695               True
```

## Failures

```text
[none]
```
