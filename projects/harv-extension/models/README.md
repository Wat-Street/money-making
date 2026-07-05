# Models

This directory is the human-facing model inventory for repo 1 and repo 2. Repo 1 and repo 2 should carry the same model set. Repo 3 may contain only draft-ready models and results.

The folders here are descriptive model names, not version buckets. Do not add `v1` or `v2` folders. If two implementations differ, name the implementation clearly, for example `CP` and `CP_REPO_FRESH`.

## Original Paper Pipeline Models

These are produced by `code/intraday_benchmark.py` and `code/daily_benchmark.py`.

| Model | Meaning |
|---|---|
| `HAR` | Baseline HAR realized-volatility model. |
| `HAR_J` | HAR with jump terms. |
| `HAR_CJ` | HAR with continuous/jump decomposition. |
| `HAR_TCJ` | HAR with threshold continuous/jump decomposition. |
| `PM` | Prime-modulo temporal-granularity model. |
| `PM_VW` | Volume-weighted PM variant. |
| `PM_AD` | Adaptive volume/stress PM variant. |
| `CP` | Older original contiguous-prime model. |
| `CP_CJ` | Contiguous-prime model with continuous/jump terms. |
| `EXH` | Exhaustive lag/window benchmark. |
| `HAM` | Hamming-code lag-position benchmark. |
| `RAND` | Random-set benchmark. |
| `CRS` | Contiguous-random-set benchmark. |

## Prop 4 CP-REPO-OPS-HG 18-Model Suite

This is the active 18-model suite used for Prop 4 testing. Results compare against `CP_REPO_FRESH`.

| Model | Meaning |
|---|---|
| `HAR_RV` | HAR-only repo RV benchmark. |
| `CP_REPO_FRESH` | Corrected fresh repo CP benchmark and main denominator. |
| `RAW_PM` | Raw PM features without HAR/CP controls. |
| `RAW_CP_REPO_PLUS_PM` | Repo CP plus raw PM features. |
| `CP_REPO_OPS_R` | Repo CP plus centered PM residue path-shape features. |
| `CP_REPO_RIDGE_OPS_R` | Ridge-shrunk OPS-R with repo CP controls unpenalized. |
| `CP_REPO_LRPM` | Local residue bridge diagnostic; unstable/non-paper in current evidence. |
| `CP_REPO_RECENT_SLOPE` | Simple recent-vs-older path-shape control. |
| `CP_REPO_HAAR_SHAPE` | Generic Haar path-shape control. |
| `CP_REPO_OPS_C` | Repo CP plus contiguous zero-sum path-shape features. |
| `CP_REPO_RIDGE_OPS_C` | Ridge-shrunk OPS-C. |
| `CP_REPO_GATED_RIDGE_OPS_C` | Gated ridge OPS-C cleaned shape model. |
| `CP_REPO_OPS_HG` | Hybrid gated OPS-C plus heavily shrunk OPS-R. |
| `RANDOM_RESIDUES_PLACEBO_REPO` | Random residue placebo. |
| `SHUFFLED_LAG_PM_PLACEBO_REPO` | Shuffled lag-label PM placebo. |
| `RANDOM_GATE_PLACEBO_REPO` | Random gate placebo. |
| `RIDGE_AR22` | Flexible AR(22) ridge benchmark. |
| `CP_REPO_OPS_K` | Optional kernel challenger in the Prop 4 suite. |

## Section 5 Robustness And Control Families

The Section 5 robustness script creates generated control families, including:

- `PM_FC{3,6,9}`, `CP_FC{3,6,9}`, `EQWIN_FC{3,6,9}`, `PREFIX_FC{3,6,9}`
- `SHIFTWIN_FC{6,9}_O{1,3,5}`
- `COMP_COPRIME_FC{6,9}`, `COMP_NONCOPRIME_FC{6,9}`
- `RAND_FC{3,6,9}_S{0,1,2}`, `CRS_FC{3,6,9}_S{0,1,2}`
- `EWMA`, `HARQ`, `GARCH11`

See `SECTION5_ROBUST_VARIANTS/` for the family-level map.

## Archived Aliases

Older stale four-block CP artifacts used non-`_REPO` aliases such as `CP_OPS_HG` and `RAW_CP_PLUS_PM`. Treat these as archived aliases, not current Prop 4 model names.
