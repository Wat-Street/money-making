# Test Results Guide

## Prop 3
Older CP+PM tests show raw CP+PM loses overall but helps cluster-entry and recent-spike conditions.

## Prop 4
The corrected 18-model repo-CP ladder does not support an always-on PM claim under headline SMAPE.

Top always-on challengers by equal-weight SMAPE advantage:

| model_name | condition | n_obs_total | n_assets | pooled_model_mean_SMAPE | equal_weight_asset_model_mean_SMAPE | pooled_CP_mean_SMAPE | equal_weight_asset_CP_mean_SMAPE | pooled_mean_advantage_vs_CP | equal_weight_asset_mean_advantage_vs_CP | pooled_median_advantage_vs_CP | equal_weight_asset_median_advantage_vs_CP | pooled_win_rate_vs_CP | equal_weight_asset_win_rate_vs_CP | pooled_winsorized_mean_advantage_1_99 | equal_weight_asset_winsorized_mean_advantage_1_99 | pooled_trimmed_mean_advantage_10pct | equal_weight_asset_trimmed_mean_advantage_10pct | pooled_mean_abs_error_model | equal_weight_asset_mean_abs_error_model | pooled_mean_abs_error_CP | equal_weight_asset_mean_abs_error_CP | pooled_mean_abs_error_advantage_vs_CP | equal_weight_asset_mean_abs_error_advantage_vs_CP | assets_positive | assets_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CP_REPO_OPS_K | all_observations | 367468 | 10 | 100.96533241845246 | 100.96608819521286 | 100.87996479236024 | 100.88077507091973 | -0.0853676260922183 | -0.0853131242931508 | 0.0024792415547641 | 0.002479767918686 | 0.4195140801375902 | 0.4195182162217158 | -0.0348048679986311 | -0.034765562463191 | 0.0165690323943487 | 0.0165825483887615 | 0.0005863111871597 | 0.0005862828821099 | 0.0005862627333686 | 0.0005862345983269 | -4.8453791123098916e-08 | -4.82837830363745e-08 | 0 | 10 |
| CP_REPO_RIDGE_OPS_R | all_observations | 367468 | 10 | 101.01518400446533 | 101.01593707597732 | 100.87996479236024 | 100.88077507091973 | -0.1352192121050798 | -0.1351620050576161 | 0.0 | 0.0 | 0.4123542730251341 | 0.4123586939635746 | -0.0789002876444404 | -0.0788568098381631 | -0.0145980788210284 | -0.0145834532211687 | 0.0005868922601175 | 0.0005868638558782 | 0.0005862627333686 | 0.0005862345983269 | -6.29526748851951e-07 | -6.29257551303435e-07 | 1 | 9 |
| CP_REPO_OPS_HG | all_observations | 367468 | 10 | 101.02476144000292 | 101.02533102028409 | 100.87996479236024 | 100.88077507091973 | -0.1447966476427205 | -0.14455594936437 | 0.0 | 0.0 | 0.4118481065017906 | 0.4118555339339663 | -0.0229204466883254 | -0.0226963399362532 | 0.0402125817807337 | 0.0403415048418169 | 0.0005815484773129 | 0.0005815197416709 | 0.0005862627333686 | 0.0005862345983269 | 4.714256055691813e-06 | 4.714856655944221e-06 | 4 | 6 |
| RANDOM_RESIDUES_PLACEBO_REPO | all_observations | 367468 | 10 | 101.03436035942428 | 101.03501972413495 | 100.87996479236024 | 100.88077507091973 | -0.1543955670640254 | -0.1542446532152538 | 2.8415676542273425e-15 | 2.8421709430404013e-15 | 0.4181071549087267 | 0.4181129298640892 | -0.0120868649659461 | -0.0119486332177825 | 0.0743293946549085 | 0.0743791737270251 | 0.0005841595452578 | 0.0005841306899137 | 0.0005862627333686 | 0.0005862345983269 | 2.103188110761372e-06 | 2.103908413149218e-06 | 2 | 8 |
| CP_REPO_GATED_RIDGE_OPS_C | all_observations | 367468 | 10 | 101.0390814503095 | 101.0397847323894 | 100.87996479236024 | 100.88077507091973 | -0.1591166579492674 | -0.1590096614697036 | 0.009367631785414 | 0.0093696206127617 | 0.4180908269563608 | 0.4180941303457799 | -0.0732666793805453 | -0.073169037393894 | 0.0104752268524412 | 0.0104955018173863 | 0.0005851564015481 | 0.0005851280965765 | 0.0005862627333686 | 0.0005862345983269 | 1.1063318205380753e-06 | 1.1065017504231826e-06 | 1 | 9 |

## Prop 5
Selective overlays are the strongest current evidence.

| model_name | strategy | mean_coverage_equal_weight | n_active_total | mean_advantage_active_rows_weighted | mean_advantage_overall_equal_weight | win_rate_overall_equal_weight | false_positive_loss_mean_equal_weight | false_negative_cost_mean_equal_weight | assets_positive_overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CP_REPO_OPS_HG | top_10pct_gate | 0.1000032645538094 | 36748 | 1.136222124777576 | 0.1136196624113833 | 0.0433070883646404 | 11.341717353331372 | 7.302660221351116 | 9 |
| CP_REPO_OPS_HG | top_5pct_gate | 0.0500016322769047 | 18374 | 2.245928068846849 | 0.1122823620831446 | 0.02285906760036 | 12.703970105069976 | 7.461461464342262 | 9 |
| RAW_CP_REPO_PLUS_PM | top_30pct_gate | 0.3000070775248609 | 110243 | 0.3017797052076316 | 0.090504905561523 | 0.129333161014889 | 4.346143923963284 | 6.253707909172006 | 9 |
| RAW_CP_REPO_PLUS_PM | top_20pct_gate | 0.2000065291076188 | 73496 | 0.4523856618289364 | 0.090451905794201 | 0.0874192661045719 | 4.442403222670338 | 6.009751855743555 | 9 |
| RAW_CP_REPO_PLUS_PM | top_10pct_gate | 0.1000032645538094 | 36748 | 0.8786548779668183 | 0.0878514748907402 | 0.0447575167508475 | 4.983900514854933 | 5.770797881935412 | 9 |

## Alternative Losses
| model_name | condition | loss_variant | loss_metric | n_obs_total | n_assets | pooled_cp_loss | pooled_model_loss | pooled_advantage_cp_minus_model | assets_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CP_REPO_RECENT_SLOPE | all_rows | standard | log_RV_MSE | 367468 | 10 | 86.01532967319302 | 85.72793855458282 | 0.2873911186101989 | 4 |
| CP_REPO_OPS_HG | all_rows | standard | log_RV_MSE | 367468 | 10 | 86.01532967319302 | 85.98216678823383 | 0.0331628849591998 | 6 |
| CP_REPO_OPS_HG | all_rows | standard | RMSE | 367468 | 10 | 0.001388809861051 | 0.0013833396482958 | 5.47021275513609e-06 | 9 |
| CP_REPO_OPS_HG | all_rows | standard | MAE | 367468 | 10 | 0.0005862627333686 | 0.000581548477313 | 4.714256055691835e-06 | 10 |
| CP_REPO_RIDGE_OPS_C | all_rows | standard | RMSE | 367468 | 10 | 0.001388809861051 | 0.0013842093680495 | 4.6004930014584665e-06 | 9 |
| CP_REPO_HAAR_SHAPE | all_rows | standard | RMSE | 367468 | 10 | 0.001388809861051 | 0.0013843120545716 | 4.497806479410366e-06 | 9 |
| RAW_CP_REPO_PLUS_PM | all_rows | standard | RMSE | 367468 | 10 | 0.001388809861051 | 0.0013855043709116 | 3.305490139376616e-06 | 10 |
| RANDOM_RESIDUES_PLACEBO_REPO | all_rows | standard | RMSE | 367468 | 10 | 0.001388809861051 | 0.0013862474850348 | 2.5623760161680234e-06 | 10 |
| CP_REPO_GATED_RIDGE_OPS_C | all_rows | standard | RMSE | 367468 | 10 | 0.001388809861051 | 0.0013864387814137 | 2.3710796373033862e-06 | 9 |
| RANDOM_RESIDUES_PLACEBO_REPO | all_rows | standard | MAE | 367468 | 10 | 0.0005862627333686 | 0.0005841595452579 | 2.1031881107614143e-06 | 9 |
| RANDOM_GATE_PLACEBO_REPO | all_rows | standard | RMSE | 367468 | 10 | 0.001388809861051 | 0.0013870116292629 | 1.7982317881185146e-06 | 9 |
| CP_REPO_HAAR_SHAPE | all_rows | standard | MAE | 367468 | 10 | 0.0005862627333686 | 0.0005845897508456 | 1.672982523003726e-06 | 9 |

## Audits
- CP reproduction passed in the corrected repo-CP run.
- Strong fast/slow passed for all ladder models on the selected 3-asset, 500-row audit.
- Feature purity passed in the recomputed audit.
- Ridge placement passed: CP/RV controls were unpenalized in ridge models.
- `CP_REPO_LRPM` is unstable and should remain diagnostic/non-paper-eligible.
- Manual conditional `n_assets` was recomputed from per-asset rows in the audit outputs.

## Research Caution
Always-on challengers lose under SMAPE. Several improve MAE/MSE/RMSE or specific regimes, so the defensible claim is selective overlay value, not global PM dominance.
