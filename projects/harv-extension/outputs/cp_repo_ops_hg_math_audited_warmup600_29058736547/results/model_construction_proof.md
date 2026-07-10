# Model Construction Proof

Generated: 2026-07-10T00:08:23.284303Z

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
 AMZN                CP_REPO_FRESH complete             13           13           False
 AMZN                       HAR_RV complete              3            3           False
 AMZN                       RAW_PM complete             10           10           False
 AMZN          RAW_CP_REPO_PLUS_PM complete             23           23           False
 AMZN                CP_REPO_OPS_R complete             23           22            True
 AMZN          CP_REPO_RIDGE_OPS_R complete             23           22            True
 AMZN                 CP_REPO_LRPM complete             38           33            True
 AMZN         CP_REPO_RECENT_SLOPE complete             18           18           False
 AMZN           CP_REPO_HAAR_SHAPE complete             31           31           False
 AMZN                CP_REPO_OPS_C complete             37           35            True
 AMZN          CP_REPO_RIDGE_OPS_C complete             37           35            True
 AMZN    CP_REPO_GATED_RIDGE_OPS_C complete             37           35            True
 AMZN               CP_REPO_OPS_HG complete             47           44            True
 AMZN RANDOM_RESIDUES_PLACEBO_REPO complete             23           22            True
 AMZN SHUFFLED_LAG_PM_PLACEBO_REPO complete             23           22            True
 AMZN     RANDOM_GATE_PLACEBO_REPO complete             37           35            True
 AMZN                   RIDGE_AR22 complete             22           22           False
 AMZN                CP_REPO_OPS_K complete             21           21           False
  EEM                CP_REPO_FRESH complete             13           13           False
  EEM                       HAR_RV complete              3            3           False
  EEM                       RAW_PM complete             10           10           False
  EEM          RAW_CP_REPO_PLUS_PM complete             23           23           False
  EEM                CP_REPO_OPS_R complete             23           22            True
  EEM          CP_REPO_RIDGE_OPS_R complete             23           22            True
  EEM                 CP_REPO_LRPM complete             38           33            True
  EEM         CP_REPO_RECENT_SLOPE complete             18           18           False
  EEM           CP_REPO_HAAR_SHAPE complete             31           31           False
  EEM                CP_REPO_OPS_C complete             37           35            True
  EEM          CP_REPO_RIDGE_OPS_C complete             37           35            True
  EEM    CP_REPO_GATED_RIDGE_OPS_C complete             37           35            True
  EEM               CP_REPO_OPS_HG complete             47           44            True
  EEM RANDOM_RESIDUES_PLACEBO_REPO complete             23           22            True
  EEM SHUFFLED_LAG_PM_PLACEBO_REPO complete             23           22            True
  EEM     RANDOM_GATE_PLACEBO_REPO complete             37           35            True
  EEM                   RIDGE_AR22 complete             22           22           False
  EEM                CP_REPO_OPS_K complete             21           21           False
  FXI                CP_REPO_FRESH complete             13           13           False
  FXI                       HAR_RV complete              3            3           False
  FXI                       RAW_PM complete             10           10           False
  FXI          RAW_CP_REPO_PLUS_PM complete             23           23           False
  FXI                CP_REPO_OPS_R complete             23           22            True
  FXI          CP_REPO_RIDGE_OPS_R complete             23           22            True
  FXI                 CP_REPO_LRPM complete             38           33            True
  FXI         CP_REPO_RECENT_SLOPE complete             18           18           False
  FXI           CP_REPO_HAAR_SHAPE complete             31           31           False
  FXI                CP_REPO_OPS_C complete             37           35            True
  FXI          CP_REPO_RIDGE_OPS_C complete             37           35            True
  FXI    CP_REPO_GATED_RIDGE_OPS_C complete             37           35            True
  FXI               CP_REPO_OPS_HG complete             47           44            True
  FXI RANDOM_RESIDUES_PLACEBO_REPO complete             23           22            True
  FXI SHUFFLED_LAG_PM_PLACEBO_REPO complete             23           22            True
  FXI     RANDOM_GATE_PLACEBO_REPO complete             37           35            True
  FXI                   RIDGE_AR22 complete             22           22           False
  FXI                CP_REPO_OPS_K complete             21           21           False
  GLD                CP_REPO_FRESH complete             13           13           False
  GLD                       HAR_RV complete              3            3           False
  GLD                       RAW_PM complete             10            8            True
  GLD          RAW_CP_REPO_PLUS_PM complete             23           21            True
  GLD                CP_REPO_OPS_R complete             23           22            True
  GLD          CP_REPO_RIDGE_OPS_R complete             23           22            True
  GLD                 CP_REPO_LRPM complete             38           33            True
  GLD         CP_REPO_RECENT_SLOPE complete             18           18           False
  GLD           CP_REPO_HAAR_SHAPE complete             31           31           False
  GLD                CP_REPO_OPS_C complete             37           35            True
  GLD          CP_REPO_RIDGE_OPS_C complete             37           35            True
  GLD    CP_REPO_GATED_RIDGE_OPS_C complete             37           35            True
  GLD               CP_REPO_OPS_HG complete             47           44            True
  GLD RANDOM_RESIDUES_PLACEBO_REPO complete             23           22            True
  GLD SHUFFLED_LAG_PM_PLACEBO_REPO complete             23           22            True
  GLD     RANDOM_GATE_PLACEBO_REPO complete             37           35            True
  GLD                   RIDGE_AR22 complete             22           22           False
  GLD                CP_REPO_OPS_K complete             21           21           False
GOOGL                CP_REPO_FRESH complete             13           13           False
GOOGL                       HAR_RV complete              3            3           False
GOOGL                       RAW_PM complete             10           10           False
GOOGL          RAW_CP_REPO_PLUS_PM complete             23           23           False
GOOGL                CP_REPO_OPS_R complete             23           22            True
GOOGL          CP_REPO_RIDGE_OPS_R complete             23           22            True
GOOGL                 CP_REPO_LRPM complete             38           33            True
GOOGL         CP_REPO_RECENT_SLOPE complete             18           18           False
GOOGL           CP_REPO_HAAR_SHAPE complete             31           31           False
GOOGL                CP_REPO_OPS_C complete             37           35            True
GOOGL          CP_REPO_RIDGE_OPS_C complete             37           35            True
GOOGL    CP_REPO_GATED_RIDGE_OPS_C complete             37           35            True
GOOGL               CP_REPO_OPS_HG complete             47           44            True
GOOGL RANDOM_RESIDUES_PLACEBO_REPO complete             23           22            True
GOOGL SHUFFLED_LAG_PM_PLACEBO_REPO complete             23           22            True
GOOGL     RANDOM_GATE_PLACEBO_REPO complete             37           35            True
GOOGL                   RIDGE_AR22 complete             22           22           False
GOOGL                CP_REPO_OPS_K complete             21           21           False
  HYG                CP_REPO_FRESH complete             13           13           False
  HYG                       HAR_RV complete              3            3           False
  HYG                       RAW_PM complete             10            8            True
  HYG          RAW_CP_REPO_PLUS_PM complete             23           21            True
  HYG                CP_REPO_OPS_R complete             23           22            True
  HYG          CP_REPO_RIDGE_OPS_R complete             23           22            True
  HYG                 CP_REPO_LRPM complete             38           33            True
  HYG         CP_REPO_RECENT_SLOPE complete             18           18           False
  HYG           CP_REPO_HAAR_SHAPE complete             31           31           False
  HYG                CP_REPO_OPS_C complete             37           35            True
  HYG          CP_REPO_RIDGE_OPS_C complete             37           35            True
  HYG    CP_REPO_GATED_RIDGE_OPS_C complete             37           35            True
  HYG               CP_REPO_OPS_HG complete             47           44            True
  HYG RANDOM_RESIDUES_PLACEBO_REPO complete             23           22            True
  HYG SHUFFLED_LAG_PM_PLACEBO_REPO complete             23           22            True
  HYG     RANDOM_GATE_PLACEBO_REPO complete             37           35            True
  HYG                   RIDGE_AR22 complete             22           22           False
  HYG                CP_REPO_OPS_K complete             21           21           False
  QQQ                CP_REPO_FRESH complete             13           13           False
  QQQ                       HAR_RV complete              3            3           False
  QQQ                       RAW_PM complete             10           10           False
  QQQ          RAW_CP_REPO_PLUS_PM complete             23           23           False
  QQQ                CP_REPO_OPS_R complete             23           22            True
  QQQ          CP_REPO_RIDGE_OPS_R complete             23           22            True
  QQQ                 CP_REPO_LRPM complete             38           33            True
  QQQ         CP_REPO_RECENT_SLOPE complete             18           18           False
  QQQ           CP_REPO_HAAR_SHAPE complete             31           31           False
  QQQ                CP_REPO_OPS_C complete             37           35            True
  QQQ          CP_REPO_RIDGE_OPS_C complete             37           35            True
  QQQ    CP_REPO_GATED_RIDGE_OPS_C complete             37           35            True
  QQQ               CP_REPO_OPS_HG complete             47           44            True
  QQQ RANDOM_RESIDUES_PLACEBO_REPO complete             23           22            True
  QQQ SHUFFLED_LAG_PM_PLACEBO_REPO complete             23           22            True
  QQQ     RANDOM_GATE_PLACEBO_REPO complete             37           35            True
  QQQ                   RIDGE_AR22 complete             22           22           False
  QQQ                CP_REPO_OPS_K complete             21           21           False
  SPY                CP_REPO_FRESH complete             13           13           False
  SPY                       HAR_RV complete              3            3           False
  SPY                       RAW_PM complete             10           10           False
  SPY          RAW_CP_REPO_PLUS_PM complete             23           23           False
  SPY                CP_REPO_OPS_R complete             23           22            True
  SPY          CP_REPO_RIDGE_OPS_R complete             23           22            True
  SPY                 CP_REPO_LRPM complete             38           33            True
  SPY         CP_REPO_RECENT_SLOPE complete             18           18           False
  SPY           CP_REPO_HAAR_SHAPE complete             31           31           False
  SPY                CP_REPO_OPS_C complete             37           35            True
  SPY          CP_REPO_RIDGE_OPS_C complete             37           35            True
  SPY    CP_REPO_GATED_RIDGE_OPS_C complete             37           35            True
  SPY               CP_REPO_OPS_HG complete             47           44            True
  SPY RANDOM_RESIDUES_PLACEBO_REPO complete             23           22            True
  SPY SHUFFLED_LAG_PM_PLACEBO_REPO complete             23           22            True
  SPY     RANDOM_GATE_PLACEBO_REPO complete             37           35            True
  SPY                   RIDGE_AR22 complete             22           22           False
  SPY                CP_REPO_OPS_K complete             21           21           False
  TLT                CP_REPO_FRESH complete             13           13           False
  TLT                       HAR_RV complete              3            3           False
  TLT                       RAW_PM complete             10           10           False
  TLT          RAW_CP_REPO_PLUS_PM complete             23           23           False
  TLT                CP_REPO_OPS_R complete             23           22            True
  TLT          CP_REPO_RIDGE_OPS_R complete             23           22            True
  TLT                 CP_REPO_LRPM complete             38           33            True
  TLT         CP_REPO_RECENT_SLOPE complete             18           18           False
  TLT           CP_REPO_HAAR_SHAPE complete             31           31           False
  TLT                CP_REPO_OPS_C complete             37           35            True
  TLT          CP_REPO_RIDGE_OPS_C complete             37           35            True
  TLT    CP_REPO_GATED_RIDGE_OPS_C complete             37           35            True
  TLT               CP_REPO_OPS_HG complete             47           44            True
  TLT RANDOM_RESIDUES_PLACEBO_REPO complete             23           22            True
  TLT SHUFFLED_LAG_PM_PLACEBO_REPO complete             23           22            True
  TLT     RANDOM_GATE_PLACEBO_REPO complete             37           35            True
  TLT                   RIDGE_AR22 complete             22           22           False
  TLT                CP_REPO_OPS_K complete             21           21           False
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
 AMZN                CP_REPO_FRESH                         True                           True
 AMZN                       HAR_RV                         True                           True
 AMZN                       RAW_PM                         True                           True
 AMZN          RAW_CP_REPO_PLUS_PM                         True                           True
 AMZN                CP_REPO_OPS_R                         True                           True
 AMZN          CP_REPO_RIDGE_OPS_R                         True                           True
 AMZN                 CP_REPO_LRPM                         True                           True
 AMZN         CP_REPO_RECENT_SLOPE                         True                           True
 AMZN           CP_REPO_HAAR_SHAPE                         True                           True
 AMZN                CP_REPO_OPS_C                         True                           True
 AMZN          CP_REPO_RIDGE_OPS_C                         True                           True
 AMZN    CP_REPO_GATED_RIDGE_OPS_C                         True                           True
 AMZN               CP_REPO_OPS_HG                         True                           True
 AMZN RANDOM_RESIDUES_PLACEBO_REPO                         True                           True
 AMZN SHUFFLED_LAG_PM_PLACEBO_REPO                         True                           True
 AMZN     RANDOM_GATE_PLACEBO_REPO                         True                           True
 AMZN                   RIDGE_AR22                         True                           True
 AMZN                CP_REPO_OPS_K                         True                           True
  EEM                CP_REPO_FRESH                         True                           True
  EEM                       HAR_RV                         True                           True
  EEM                       RAW_PM                         True                           True
  EEM          RAW_CP_REPO_PLUS_PM                         True                           True
  EEM                CP_REPO_OPS_R                         True                           True
  EEM          CP_REPO_RIDGE_OPS_R                         True                           True
  EEM                 CP_REPO_LRPM                         True                           True
  EEM         CP_REPO_RECENT_SLOPE                         True                           True
  EEM           CP_REPO_HAAR_SHAPE                         True                           True
  EEM                CP_REPO_OPS_C                         True                           True
  EEM          CP_REPO_RIDGE_OPS_C                         True                           True
  EEM    CP_REPO_GATED_RIDGE_OPS_C                         True                           True
  EEM               CP_REPO_OPS_HG                         True                           True
  EEM RANDOM_RESIDUES_PLACEBO_REPO                         True                           True
  EEM SHUFFLED_LAG_PM_PLACEBO_REPO                         True                           True
  EEM     RANDOM_GATE_PLACEBO_REPO                         True                           True
  EEM                   RIDGE_AR22                         True                           True
  EEM                CP_REPO_OPS_K                         True                           True
  FXI                CP_REPO_FRESH                         True                           True
  FXI                       HAR_RV                         True                           True
  FXI                       RAW_PM                         True                           True
  FXI          RAW_CP_REPO_PLUS_PM                         True                           True
  FXI                CP_REPO_OPS_R                         True                           True
  FXI          CP_REPO_RIDGE_OPS_R                         True                           True
  FXI                 CP_REPO_LRPM                         True                           True
  FXI         CP_REPO_RECENT_SLOPE                         True                           True
  FXI           CP_REPO_HAAR_SHAPE                         True                           True
  FXI                CP_REPO_OPS_C                         True                           True
  FXI          CP_REPO_RIDGE_OPS_C                         True                           True
  FXI    CP_REPO_GATED_RIDGE_OPS_C                         True                           True
  FXI               CP_REPO_OPS_HG                         True                           True
  FXI RANDOM_RESIDUES_PLACEBO_REPO                         True                           True
  FXI SHUFFLED_LAG_PM_PLACEBO_REPO                         True                           True
  FXI     RANDOM_GATE_PLACEBO_REPO                         True                           True
  FXI                   RIDGE_AR22                         True                           True
  FXI                CP_REPO_OPS_K                         True                           True
  GLD                CP_REPO_FRESH                         True                           True
  GLD                       HAR_RV                         True                           True
  GLD                       RAW_PM                         True                           True
  GLD          RAW_CP_REPO_PLUS_PM                         True                           True
  GLD                CP_REPO_OPS_R                         True                           True
  GLD          CP_REPO_RIDGE_OPS_R                         True                           True
  GLD                 CP_REPO_LRPM                         True                           True
  GLD         CP_REPO_RECENT_SLOPE                         True                           True
  GLD           CP_REPO_HAAR_SHAPE                         True                           True
  GLD                CP_REPO_OPS_C                         True                           True
  GLD          CP_REPO_RIDGE_OPS_C                         True                           True
  GLD    CP_REPO_GATED_RIDGE_OPS_C                         True                           True
  GLD               CP_REPO_OPS_HG                         True                           True
  GLD RANDOM_RESIDUES_PLACEBO_REPO                         True                           True
  GLD SHUFFLED_LAG_PM_PLACEBO_REPO                         True                           True
  GLD     RANDOM_GATE_PLACEBO_REPO                         True                           True
  GLD                   RIDGE_AR22                         True                           True
  GLD                CP_REPO_OPS_K                         True                           True
GOOGL                CP_REPO_FRESH                         True                           True
GOOGL                       HAR_RV                         True                           True
GOOGL                       RAW_PM                         True                           True
GOOGL          RAW_CP_REPO_PLUS_PM                         True                           True
GOOGL                CP_REPO_OPS_R                         True                           True
GOOGL          CP_REPO_RIDGE_OPS_R                         True                           True
GOOGL                 CP_REPO_LRPM                         True                           True
GOOGL         CP_REPO_RECENT_SLOPE                         True                           True
GOOGL           CP_REPO_HAAR_SHAPE                         True                           True
GOOGL                CP_REPO_OPS_C                         True                           True
GOOGL          CP_REPO_RIDGE_OPS_C                         True                           True
GOOGL    CP_REPO_GATED_RIDGE_OPS_C                         True                           True
GOOGL               CP_REPO_OPS_HG                         True                           True
GOOGL RANDOM_RESIDUES_PLACEBO_REPO                         True                           True
GOOGL SHUFFLED_LAG_PM_PLACEBO_REPO                         True                           True
GOOGL     RANDOM_GATE_PLACEBO_REPO                         True                           True
GOOGL                   RIDGE_AR22                         True                           True
GOOGL                CP_REPO_OPS_K                         True                           True
  HYG                CP_REPO_FRESH                         True                           True
  HYG                       HAR_RV                         True                           True
  HYG                       RAW_PM                         True                           True
  HYG          RAW_CP_REPO_PLUS_PM                         True                           True
  HYG                CP_REPO_OPS_R                         True                           True
  HYG          CP_REPO_RIDGE_OPS_R                         True                           True
  HYG                 CP_REPO_LRPM                         True                           True
  HYG         CP_REPO_RECENT_SLOPE                         True                           True
  HYG           CP_REPO_HAAR_SHAPE                         True                           True
  HYG                CP_REPO_OPS_C                         True                           True
  HYG          CP_REPO_RIDGE_OPS_C                         True                           True
  HYG    CP_REPO_GATED_RIDGE_OPS_C                         True                           True
  HYG               CP_REPO_OPS_HG                         True                           True
  HYG RANDOM_RESIDUES_PLACEBO_REPO                         True                           True
  HYG SHUFFLED_LAG_PM_PLACEBO_REPO                         True                           True
  HYG     RANDOM_GATE_PLACEBO_REPO                         True                           True
  HYG                   RIDGE_AR22                         True                           True
  HYG                CP_REPO_OPS_K                         True                           True
  QQQ                CP_REPO_FRESH                         True                           True
  QQQ                       HAR_RV                         True                           True
  QQQ                       RAW_PM                         True                           True
  QQQ          RAW_CP_REPO_PLUS_PM                         True                           True
  QQQ                CP_REPO_OPS_R                         True                           True
  QQQ          CP_REPO_RIDGE_OPS_R                         True                           True
  QQQ                 CP_REPO_LRPM                         True                           True
  QQQ         CP_REPO_RECENT_SLOPE                         True                           True
  QQQ           CP_REPO_HAAR_SHAPE                         True                           True
  QQQ                CP_REPO_OPS_C                         True                           True
  QQQ          CP_REPO_RIDGE_OPS_C                         True                           True
  QQQ    CP_REPO_GATED_RIDGE_OPS_C                         True                           True
  QQQ               CP_REPO_OPS_HG                         True                           True
  QQQ RANDOM_RESIDUES_PLACEBO_REPO                         True                           True
  QQQ SHUFFLED_LAG_PM_PLACEBO_REPO                         True                           True
  QQQ     RANDOM_GATE_PLACEBO_REPO                         True                           True
  QQQ                   RIDGE_AR22                         True                           True
  QQQ                CP_REPO_OPS_K                         True                           True
  SPY                CP_REPO_FRESH                         True                           True
  SPY                       HAR_RV                         True                           True
  SPY                       RAW_PM                         True                           True
  SPY          RAW_CP_REPO_PLUS_PM                         True                           True
  SPY                CP_REPO_OPS_R                         True                           True
  SPY          CP_REPO_RIDGE_OPS_R                         True                           True
  SPY                 CP_REPO_LRPM                         True                           True
  SPY         CP_REPO_RECENT_SLOPE                         True                           True
  SPY           CP_REPO_HAAR_SHAPE                         True                           True
  SPY                CP_REPO_OPS_C                         True                           True
  SPY          CP_REPO_RIDGE_OPS_C                         True                           True
  SPY    CP_REPO_GATED_RIDGE_OPS_C                         True                           True
  SPY               CP_REPO_OPS_HG                         True                           True
  SPY RANDOM_RESIDUES_PLACEBO_REPO                         True                           True
  SPY SHUFFLED_LAG_PM_PLACEBO_REPO                         True                           True
  SPY     RANDOM_GATE_PLACEBO_REPO                         True                           True
  SPY                   RIDGE_AR22                         True                           True
  SPY                CP_REPO_OPS_K                         True                           True
  TLT                CP_REPO_FRESH                         True                           True
  TLT                       HAR_RV                         True                           True
  TLT                       RAW_PM                         True                           True
  TLT          RAW_CP_REPO_PLUS_PM                         True                           True
  TLT                CP_REPO_OPS_R                         True                           True
  TLT          CP_REPO_RIDGE_OPS_R                         True                           True
  TLT                 CP_REPO_LRPM                         True                           True
  TLT         CP_REPO_RECENT_SLOPE                         True                           True
  TLT           CP_REPO_HAAR_SHAPE                         True                           True
  TLT                CP_REPO_OPS_C                         True                           True
  TLT          CP_REPO_RIDGE_OPS_C                         True                           True
  TLT    CP_REPO_GATED_RIDGE_OPS_C                         True                           True
  TLT               CP_REPO_OPS_HG                         True                           True
  TLT RANDOM_RESIDUES_PLACEBO_REPO                         True                           True
  TLT SHUFFLED_LAG_PM_PLACEBO_REPO                         True                           True
  TLT     RANDOM_GATE_PLACEBO_REPO                         True                           True
  TLT                   RIDGE_AR22                         True                           True
  TLT                CP_REPO_OPS_K                         True                           True
```

## Failure Status

```text
[none]
```
