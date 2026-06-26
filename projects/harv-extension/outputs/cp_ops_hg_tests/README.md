# CP-OPS-HG Tests

Generated: 2026-06-26T18:07:49.115669Z

## Run

- Mode: `smoke`
- Phase: `all`
- Assets: `AAPL`
- Models requested: `CP_FRESH,HAR_RV,RAW_PM,RAW_CP_PLUS_PM,CP_OPS_R,CP_OPS_C,CP_RIDGE_OPS_C,CP_GATED_RIDGE_OPS_C,CP_RIDGE_OPS_R,CP_LRPM,CP_RECENT_SLOPE,CP_HAAR_SHAPE,CP_OPS_HG,RANDOM_RESIDUES_PLACEBO,SHUFFLED_LAG_PM_PLACEBO,RANDOM_GATE_PLACEBO,RIDGE_AR22,CP_OPS_K`
- Seed: `42`
- Reuse prior baselines: `False`
- Skip existing: `False`
- Force rerun: `True`
- Command: `outputs\cp_ops_hg_tests\scripts\run_cp_ops_hg_tests.py --mode smoke --assets AAPL --phase all --seed 42 --force`
- Paper eligibility rule: rows are paper eligible only when both the model and CP benchmark were generated in the current full run. Smoke and reused-prior rows are excluded.

## Compatibility Note

The repository CP benchmark is `contig_prime_modulo`, not an explicit disjoint CP-block projection. This runner preserves that CP benchmark for `CP_FRESH` and raw CP+PM reproduction, while OPS/shape zero-sum checks use the locked fallback blocks `[[1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21, 22]]`.

## Commands

Smoke test:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode smoke --assets AAPL --phase all --seed 42 --force
```

Phase 1 run:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode full --phase phase1 --seed 42 --skip-existing
```

Phase 2 / full linear ladder:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode full --phase linear --seed 42 --skip-existing
```

Final all-model full run:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode full --phase all --seed 42 --skip-existing
```

Regenerate paper tables from existing outputs:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode tables-only --outdir outputs/cp_ops_hg_tests
```

## Interpretation Rules

- If gated OPS-C beats CP, temporal path shape matters after CP.
- If CP-OPS-HG beats gated OPS-C, original PM residue structure adds value beyond local shape.
- If OPS-R fails against random/shuffled residues, PM-specific residue structure is not supported.
- If Haar or recent slope matches OPS-C, the result is generic path shape rather than PM-specific.
- If improvements are mostly positive mean advantage with sub-50% win rate, describe them as gain-size effects, not dominance.
- If exit/high-dispersion losses disappear, the model has solved the raw PM noise problem.
- If CP-OPS-HG loses to gated OPS-C, OPS-R is contaminating the final hybrid and should be reported as rejected.

## Zero-Block-Sum Check

Shape feature failures: `0`

## Ablation Preview

```text
             model_name        condition  n_obs  mean_advantage_vs_CP  win_rate_vs_CP  is_paper_eligible
                 HAR_RV all_observations     98             -2.976477        0.357143              False
               CP_FRESH all_observations     98              0.000000        0.000000              False
                 RAW_PM all_observations     98             -2.879569        0.459184              False
         RAW_CP_PLUS_PM all_observations     98             -0.736406        0.500000              False
               CP_OPS_R all_observations     98             -0.626109        0.438776              False
         CP_RIDGE_OPS_R all_observations     98             -0.626114        0.428571              False
                CP_LRPM all_observations     98              0.612584        0.510204              False
        CP_RECENT_SLOPE all_observations     98              0.006457        0.448980              False
          CP_HAAR_SHAPE all_observations     98             -0.571685        0.448980              False
               CP_OPS_C all_observations     98             -0.955807        0.448980              False
         CP_RIDGE_OPS_C all_observations     98             -0.955646        0.459184              False
   CP_GATED_RIDGE_OPS_C all_observations     98             -2.541183        0.438776              False
              CP_OPS_HG all_observations     98             -4.065253        0.448980              False
RANDOM_RESIDUES_PLACEBO all_observations     98              1.099294        0.520408              False
SHUFFLED_LAG_PM_PLACEBO all_observations     98             -0.314010        0.448980              False
    RANDOM_GATE_PLACEBO all_observations     98             -1.864271        0.418367              False
             RIDGE_AR22 all_observations     98             -4.994254        0.367347              False
               CP_OPS_K all_observations     98              0.387264        0.479592              False
```

## Failures

```text
[none]
```
