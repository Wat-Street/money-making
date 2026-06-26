# CP-OPS-HG Tests

Generated: 2026-06-26T19:05:27.949912Z

## Run

- Mode: `tables-only`
- Phase: `all`
- Assets: `AAPL`
- Models requested: `CP_FRESH,HAR_RV,RAW_PM,RAW_CP_PLUS_PM,CP_OPS_R,CP_OPS_C,CP_RIDGE_OPS_C,CP_GATED_RIDGE_OPS_C,CP_RIDGE_OPS_R,CP_LRPM,CP_RECENT_SLOPE,CP_HAAR_SHAPE,CP_OPS_HG,RANDOM_RESIDUES_PLACEBO,SHUFFLED_LAG_PM_PLACEBO,RANDOM_GATE_PLACEBO,RIDGE_AR22,CP_OPS_K`
- Seed: `42`
- Reuse prior baselines: `False`
- Skip existing: `False`
- Force rerun: `False`
- Target transform: `level`
- Ridge lambda grid: `0.001`
- Ridge R-ratio grid: `10`
- CV mode: `auto`
- Command: `outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode tables-only --assets AAPL --phase all --seed 42`
- Paper eligibility rule: rows are paper eligible only when both the model and CP benchmark were generated in the current full run and exact timestamp/actual alignment passes. Smoke, reused-prior, skipped, and scaffolded rows are excluded.

## Locked CP Benchmark

`CP_FRESH` is the explicit disjoint-block CP benchmark, not the older repository `contig_prime_modulo` model. For `n=22`, the locked CP blocks are `[[1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20, 21, 22]]`. OPS-R, OPS-C, LRPM, recent slope, Haar, KOPS, and placebo shape features are all zero-sum against this same CP block space.

## Commands

Smoke test:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode smoke --assets AAPL --phase all --seed 42 --force --audit-no-lookahead
```

Phase 1 run:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode full --phase phase1 --seed 42 --skip-existing --cv-mode inner --lambda-grid 0.0001,0.001,0.01
```

Phase 2 / full linear ladder:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode full --phase linear --seed 42 --skip-existing --cv-mode inner --lambda-grid 0.0001,0.001,0.01 --lambda-r-ratio-grid 2,5,10,20
```

Final all-model full run:

```powershell
python outputs/cp_ops_hg_tests/scripts/run_cp_ops_hg_tests.py --mode full --phase all --seed 42 --skip-existing --audit-no-lookahead --cv-mode inner --lambda-grid 0.0001,0.001,0.01 --lambda-r-ratio-grid 2,5,10,20
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

## Research Conclusion Guardrail

- If gated OPS-C beats CP but OPS-R adds nothing, the result validates temporal path shape, not original PM residue specificity.
- If OPS-R beats random/shuffled residues and adds beyond OPS-C, PM-specific residue structure is supported.
- If Haar/recent slope matches OPS-C, the result is generic local shape, not PM-specific.
- If positive mean advantage occurs with win rate below 50%, describe it as a gain-size effect, not dominance.
- If CP-OPS-HG loses to gated OPS-C, OPS-R is contaminating the final hybrid and should be rejected.

## Zero-Block-Sum Check

Shape feature failures: `0`

## Ablation Preview

```text
             model_name        condition  n_obs  mean_advantage_vs_CP  win_rate_vs_CP  is_paper_eligible
                 HAR_RV all_observations     98             -1.744497        0.448980              False
               CP_FRESH all_observations     98              0.000000        0.000000              False
                 RAW_PM all_observations     98             -2.385255        0.367347              False
         RAW_CP_PLUS_PM all_observations     98             -0.441241        0.500000              False
               CP_OPS_R all_observations     98              0.105170        0.540816              False
         CP_RIDGE_OPS_R all_observations     98              0.104443        0.530612              False
                CP_LRPM all_observations     98              2.433814        0.612245              False
        CP_RECENT_SLOPE all_observations     98              0.020423        0.520408              False
          CP_HAAR_SHAPE all_observations     98              0.490568        0.479592              False
               CP_OPS_C all_observations     98              0.720682        0.500000              False
         CP_RIDGE_OPS_C all_observations     98              0.720411        0.510204              False
   CP_GATED_RIDGE_OPS_C all_observations     98             -1.261847        0.500000              False
              CP_OPS_HG all_observations     98             -0.288908        0.520408              False
RANDOM_RESIDUES_PLACEBO all_observations     98              0.937754        0.530612              False
SHUFFLED_LAG_PM_PLACEBO all_observations     98             -0.359767        0.448980              False
    RANDOM_GATE_PLACEBO all_observations     98              1.569289        0.500000              False
             RIDGE_AR22 all_observations     98             -1.693133        0.326531              False
               CP_OPS_K all_observations     98              0.213970        0.530612              False
```

## Failures

```text
[none]
```
