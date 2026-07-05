# Model Construction Proof

Generated: 2026-07-05T07:43:23.001451Z

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
asset    model_name   status  feature_count  design_rank  rank_deficient
 AAPL CP_REPO_FRESH complete             13           13           False
 AAPL        PM_QDK complete             71           71           False
 AAPL      PM_QDK_2 complete            134          134           False
 AAPL          PHQO complete             22           22           False
 AMZN CP_REPO_FRESH complete             13           13           False
 AMZN        PM_QDK complete             71           71           False
 AMZN      PM_QDK_2 complete            134          134           False
 AMZN          PHQO complete             22           22           False
  EEM CP_REPO_FRESH complete             13           13           False
  EEM        PM_QDK complete             71           71           False
  EEM      PM_QDK_2 complete            134          134           False
  EEM          PHQO complete             22           22           False
  FXI CP_REPO_FRESH complete             13           13           False
  FXI        PM_QDK complete             71           71           False
  FXI      PM_QDK_2 complete            134          134           False
  FXI          PHQO complete             22           22           False
  GLD CP_REPO_FRESH complete             13           13           False
  GLD        PM_QDK complete             71           71           False
  GLD      PM_QDK_2 complete            134          134           False
  GLD          PHQO complete             22           22           False
GOOGL CP_REPO_FRESH complete             13           13           False
GOOGL        PM_QDK complete             71           71           False
GOOGL      PM_QDK_2 complete            134          134           False
GOOGL          PHQO complete             22           22           False
  HYG CP_REPO_FRESH complete             13           13           False
  HYG        PM_QDK complete             71           71           False
  HYG      PM_QDK_2 complete            134          134           False
  HYG          PHQO complete             22           22           False
  QQQ CP_REPO_FRESH complete             13           13           False
  QQQ        PM_QDK complete             71           71           False
  QQQ      PM_QDK_2 complete            134          134           False
  QQQ          PHQO complete             22           22           False
  SPY CP_REPO_FRESH complete             13           13           False
  SPY        PM_QDK complete             71           71           False
  SPY      PM_QDK_2 complete            134          134           False
  SPY          PHQO complete             22           22           False
  TLT CP_REPO_FRESH complete             13           13           False
  TLT        PM_QDK complete             71           71           False
  TLT      PM_QDK_2 complete            134          134           False
  TLT          PHQO complete             22           22           False
```

## Alignment Proof

Every model comparison is paired to `CP_REPO_FRESH` by timestamp and target value before metrics are computed. Paper eligibility requires exact timestamp and actual-target matches.

```text
asset    model_name  timestamps_exactly_match_CP  actual_target_values_match_CP
 AAPL CP_REPO_FRESH                         True                           True
 AAPL        PM_QDK                         True                           True
 AAPL      PM_QDK_2                         True                           True
 AAPL          PHQO                         True                           True
 AMZN CP_REPO_FRESH                         True                           True
 AMZN        PM_QDK                         True                           True
 AMZN      PM_QDK_2                         True                           True
 AMZN          PHQO                         True                           True
  EEM CP_REPO_FRESH                         True                           True
  EEM        PM_QDK                         True                           True
  EEM      PM_QDK_2                         True                           True
  EEM          PHQO                        False                           True
  FXI CP_REPO_FRESH                         True                           True
  FXI        PM_QDK                         True                           True
  FXI      PM_QDK_2                         True                           True
  FXI          PHQO                         True                           True
  GLD CP_REPO_FRESH                         True                           True
  GLD        PM_QDK                         True                           True
  GLD      PM_QDK_2                         True                           True
  GLD          PHQO                        False                           True
GOOGL CP_REPO_FRESH                         True                           True
GOOGL        PM_QDK                         True                           True
GOOGL      PM_QDK_2                         True                           True
GOOGL          PHQO                         True                           True
  HYG CP_REPO_FRESH                         True                           True
  HYG        PM_QDK                         True                           True
  HYG      PM_QDK_2                         True                           True
  HYG          PHQO                        False                           True
  QQQ CP_REPO_FRESH                         True                           True
  QQQ        PM_QDK                         True                           True
  QQQ      PM_QDK_2                         True                           True
  QQQ          PHQO                         True                           True
  SPY CP_REPO_FRESH                         True                           True
  SPY        PM_QDK                         True                           True
  SPY      PM_QDK_2                         True                           True
  SPY          PHQO                         True                           True
  TLT CP_REPO_FRESH                         True                           True
  TLT        PM_QDK                         True                           True
  TLT      PM_QDK_2                         True                           True
  TLT          PHQO                         True                           True
```

## Failure Status

```text
[none]
```
