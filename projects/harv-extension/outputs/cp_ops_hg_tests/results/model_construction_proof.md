# Model Construction Proof

Generated: 2026-06-26T19:05:27.944006Z

## CP Block Definition

The paper benchmark `CP_FRESH` uses the explicit disjoint block averages:

```text
B1 = [1, 2, 3]
B2 = [4, 5, 6, 7, 8]
B3 = [9, 10, 11, 12, 13, 14]
B4 = [15, 16, 17, 18, 19, 20, 21, 22]
```

The CP projection `Pi_CP` maps the 22-lag vector to its within-block means. Every OPS/shape family is built in `(I - Pi_CP)x_t` by subtracting each block's average from the candidate lag-weight vector.

## Zero-Sum Proof

For every generated OPS-R, LRPM, recent-slope, Haar, OPS-C, random-residue, shuffled-PM, and KOPS feature, the manifest evaluates `sum_{ell in B_j} w_ell` for every CP block. The maximum tolerated absolute block sum is `1e-10`. Manifest failures: `0`.

OPS-R is `(I - Pi_CP)u_(p,r)`. OPS-C uses adjacent sub-block contrasts inside each CP block. KOPS uses centered prime-signature kernel eigenvectors inside each CP block. Rank dependencies are handled by ridge penalties where registered and by SVD pseudo-inverse fallback in the expanding solver.

## Feature Counts And Design Ranks

```text
asset              model_name   status  feature_count  design_rank  rank_deficient
 AAPL                CP_FRESH complete              4            4           False
 AAPL                  HAR_RV complete              3            3           False
 AAPL                  RAW_PM complete             10           10           False
 AAPL          RAW_CP_PLUS_PM complete             14           14           False
 AAPL                CP_OPS_R complete             14           13            True
 AAPL                CP_OPS_C complete             28           26            True
 AAPL          CP_RIDGE_OPS_C complete             28           26            True
 AAPL    CP_GATED_RIDGE_OPS_C complete             28           26            True
 AAPL          CP_RIDGE_OPS_R complete             14           13            True
 AAPL                 CP_LRPM complete             29           24            True
 AAPL         CP_RECENT_SLOPE complete              6            6           False
 AAPL           CP_HAAR_SHAPE complete             22           22           False
 AAPL               CP_OPS_HG complete             38           35            True
 AAPL RANDOM_RESIDUES_PLACEBO complete             14           13            True
 AAPL SHUFFLED_LAG_PM_PLACEBO complete             14           13            True
 AAPL     RANDOM_GATE_PLACEBO complete             28           26            True
 AAPL              RIDGE_AR22 complete             22           22           False
 AAPL                CP_OPS_K complete             12           12           False
```

## Alignment Proof

Every model comparison is paired to `CP_FRESH` by timestamp and target value before metrics are computed. Paper eligibility requires exact timestamp and actual-target matches.

```text
asset              model_name  timestamps_exactly_match_CP  actual_target_values_match_CP
 AAPL                CP_FRESH                         True                           True
 AAPL                  HAR_RV                         True                           True
 AAPL                  RAW_PM                         True                           True
 AAPL          RAW_CP_PLUS_PM                         True                           True
 AAPL                CP_OPS_R                         True                           True
 AAPL                CP_OPS_C                         True                           True
 AAPL          CP_RIDGE_OPS_C                         True                           True
 AAPL    CP_GATED_RIDGE_OPS_C                         True                           True
 AAPL          CP_RIDGE_OPS_R                         True                           True
 AAPL                 CP_LRPM                         True                           True
 AAPL         CP_RECENT_SLOPE                         True                           True
 AAPL           CP_HAAR_SHAPE                         True                           True
 AAPL               CP_OPS_HG                         True                           True
 AAPL RANDOM_RESIDUES_PLACEBO                         True                           True
 AAPL SHUFFLED_LAG_PM_PLACEBO                         True                           True
 AAPL     RANDOM_GATE_PLACEBO                         True                           True
 AAPL              RIDGE_AR22                         True                           True
 AAPL                CP_OPS_K                         True                           True
```

## Failure Status

```text
[none]
```
