# Model Construction Proof

Generated: 2026-07-09T23:24:14.093052Z

## CP Block Definition

The paper benchmark `CP_REPO_FRESH` uses the repository `contig_prime_modulo` CP design: all `RV*` and `CP_*` columns selected with the same convention as the original CP+PM incremental script.

The four lag zones below are retained only for local shape diagnostics and feature construction scaffolds, not as the primary CP benchmark:

```text
B1 = [1, 2, 3]
B2 = [4, 5, 6, 7, 8]
B3 = [9, 10, 11, 12, 13, 14]
B4 = [15, 16, 17, 18, 19, 20, 21, 22]
```

Incremental shape models use `y_(t+1) = alpha + beta'C_t + theta'Z_t + error`, where `C_t` is the repo CP design. Ridge penalties, when used, apply to `Z_t` only, not to the intercept or repo CP controls.

## Level-Removal Diagnostics

For generated shape weights, the manifest reports both diagnostic lag-zone block sums and the global lag-weight sum. The hard diagnostic is global level removal with tolerance `1e-10`. Global level-removal failures: `0`.

OPS-R is implemented as centered within-prime residue contrasts (`PMCTR`). OPS-C uses adjacent local sub-block contrasts in lag zones. KOPS uses centered prime-signature kernel features in lag zones. The claim is incremental value after exact repo CP controls, not projection orthogonality to repo CP.

## Feature Counts And Design Ranks

```text
asset                   model_name   status  feature_count  design_rank  rank_deficient
 AAPL                CP_REPO_FRESH complete             13           13           False
 AAPL                       HAR_RV complete              3            3           False
 AAPL                       RAW_PM complete             10           10           False
 AAPL          RAW_CP_REPO_PLUS_PM complete             23           23           False
 AAPL                CP_REPO_OPS_R complete             23           22            True
 AAPL          CP_REPO_RIDGE_OPS_R complete             23           22            True
 AAPL                 CP_REPO_LRPM complete             38           33            True
 AAPL         CP_REPO_RECENT_SLOPE complete             18           18           False
 AAPL           CP_REPO_HAAR_SHAPE complete             31           31           False
 AAPL                CP_REPO_OPS_C complete             37           35            True
 AAPL          CP_REPO_RIDGE_OPS_C complete             37           35            True
 AAPL    CP_REPO_GATED_RIDGE_OPS_C complete             37           35            True
 AAPL               CP_REPO_OPS_HG complete             47           44            True
 AAPL RANDOM_RESIDUES_PLACEBO_REPO complete             23           22            True
 AAPL SHUFFLED_LAG_PM_PLACEBO_REPO complete             23           22            True
 AAPL     RANDOM_GATE_PLACEBO_REPO complete             37           35            True
 AAPL                   RIDGE_AR22 complete             22           22           False
 AAPL                CP_REPO_OPS_K complete             21           21           False
```

## Alignment Proof

Every model comparison is paired to `CP_REPO_FRESH` by timestamp and target value before metrics are computed. Paper eligibility requires exact timestamp and actual-target matches.

```text
asset                   model_name  timestamps_exactly_match_CP  actual_target_values_match_CP
 AAPL                CP_REPO_FRESH                         True                           True
 AAPL                       HAR_RV                         True                           True
 AAPL                       RAW_PM                         True                           True
 AAPL          RAW_CP_REPO_PLUS_PM                         True                           True
 AAPL                CP_REPO_OPS_R                         True                           True
 AAPL          CP_REPO_RIDGE_OPS_R                         True                           True
 AAPL                 CP_REPO_LRPM                         True                           True
 AAPL         CP_REPO_RECENT_SLOPE                         True                           True
 AAPL           CP_REPO_HAAR_SHAPE                         True                           True
 AAPL                CP_REPO_OPS_C                         True                           True
 AAPL          CP_REPO_RIDGE_OPS_C                         True                           True
 AAPL    CP_REPO_GATED_RIDGE_OPS_C                         True                           True
 AAPL               CP_REPO_OPS_HG                         True                           True
 AAPL RANDOM_RESIDUES_PLACEBO_REPO                         True                           True
 AAPL SHUFFLED_LAG_PM_PLACEBO_REPO                         True                           True
 AAPL     RANDOM_GATE_PLACEBO_REPO                         True                           True
 AAPL                   RIDGE_AR22                         True                           True
 AAPL                CP_REPO_OPS_K                         True                           True
```

## Failure Status

```text
[none]
```
