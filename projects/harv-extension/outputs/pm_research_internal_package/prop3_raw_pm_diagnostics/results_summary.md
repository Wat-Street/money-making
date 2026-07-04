# Prop 3 Results Summary

Known result pattern: all observations negative, cluster-entry positive, recent-spike positive, cluster-exit negative, older-spike negative.

| condition | total_n_obs | number_of_assets | equal_weight_asset_mean_advantage | equal_weight_asset_hybrid_win_rate | number_of_assets_where_hybrid_beats_CP |
| --- | --- | --- | --- | --- | --- |
| all_observations | 367248 | 10 | -0.2152246431185233 | 0.407793726347625 | 2 |
| cluster_entry_loose | 73457 | 10 | 0.2463551786205983 | 0.4494545950696508 | 9 |
| cluster_entry_strict | 73266 | 10 | 0.244475745365562 | 0.4496442387409313 | 9 |
| cluster_exit_loose | 73455 | 10 | -0.797489704267947 | 0.3850850676820078 | 0 |
| cluster_exit_strict | 73334 | 10 | -0.8014896725671911 | 0.385224024320799 | 0 |
| inflection_points | 144932 | 10 | -0.2893022935178156 | 0.4249447797508324 | 0 |
| high_within_block_dispersion | 68970 | 10 | 0.0608970673303894 | 0.3785312910123724 | 6 |
| recent_spike_position | 124971 | 10 | 0.2380825492861734 | 0.4234141995198976 | 8 |
| older_spike_position | 109235 | 10 | -0.4466940662161353 | 0.3997001266438508 | 0 |
| same_average_different_path | 7396 | 10 | -0.1188882311986267 | 0.3799496595697643 | 3 |

Interpretation: raw/global PM is noisy overall but has conditional signal in entry/recent-spike regimes.
