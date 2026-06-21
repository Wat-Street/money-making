# Conditional PM-vs-CP Test Report

## Overview

This directory contains a **standalone, read-only conditional PM-vs-CP performance test** using timestamp-level predictions from the main HAR-V extension benchmark.

**Goal**: Test whether PM's temporal granularity adds value exactly when temporal order should matter most—specifically during volatility cluster transitions, inflection points, and high within-block dispersion.

## Methodology

### Data Source

- **Predictions**: `run_results/current_intraday/predictions/*.csv`
- **Tickers**: 10 (AAPL, AMZN, EEM, FXI, GLD, GOOGL, HYG, QQQ, SPY, TLT)
- **Total observations**: ~367,000 timestamp-level forecasts
- **Features per observation**: 
  - Date (timestamp)
  - Actual (realized volatility target)
  - Predicted_PM, Predicted_CP (model forecasts)
  - SMAPE_PM_pct, SMAPE_CP_pct (forecast errors)
  - AbsErr_PM, AbsErr_CP (absolute errors)

### Anti-Lookahead Rule (Critical)

**All condition flags are constructed using lagged Actual RV values only.**

- No future information is used.
- For each observation at time $t$, all condition flags are defined using `Actual.shift(k)` for $k \geq 1$.
- This ensures the forecast error at time $t$ depends only on information available before the forecast.
- Example: `recent_mean_1_5 = mean(lag1, lag2, ..., lag5)` uses only past volatility.

### Feature Engineering

1. **Lag construction**: For each asset, we create lagged RV columns:
   ```
   lag1 = Actual.shift(1)
   lag2 = Actual.shift(2)
   ...
   lag22 = Actual.shift(22)
   ```
   After creating lags, rows with NaN values are dropped (first 22 observations per asset).

2. **PM advantage metric**:
   ```
   PM_advantage_vs_CP = SMAPE_CP_pct - SMAPE_PM_pct
   ```
   - Positive = PM beats CP (lower SMAPE%)
   - Negative = CP beats PM

### Condition Definitions (8 Subsets)

Each condition partitions observations based on lagged volatility patterns:

#### 1. **all_observations**
- Baseline: all valid rows after lag creation.
- Interpretation: overall unconditional performance.

#### 2. & 3. **cluster_entry_loose / cluster_entry_strict**
- **Loose**: Top 20% of entry score.
  - `recent_mean_1_5 = mean(lag1…lag5)`
  - `older_mean_6_10 = mean(lag6…lag10)`
  - `entry_score = recent_mean_1_5 - older_mean_6_10`
  - Flag: `entry_score >= 80th percentile`
  
- **Strict**: Recent lags above asset median + loose criterion.
  - Flag: `(recent_mean_1_5 >= median_lagged_RV) & loose`

- **Interpretation**: Rising volatility regime. PM's order sensitivity should shine if microstructure matters.

#### 4. & 5. **cluster_exit_loose / cluster_exit_strict**
- **Loose**: Top 20% of exit score.
  - `exit_score = older_mean_6_10 - recent_mean_1_5`
  - Flag: `exit_score >= 80th percentile`
  
- **Strict**: Older lags above asset median + loose criterion.
  - Flag: `(older_mean_6_10 >= median_lagged_RV) & loose`

- **Interpretation**: Falling volatility regime. CP's contiguous averaging might smooth away important order signals.

#### 6. **inflection_points**
- Local volatility direction changes.
- `slope_recent = mean(lag1…lag3) - mean(lag4…lag6)`
- `slope_prior = mean(lag4…lag6) - mean(lag7…lag10)`
- Flag: `(slope_recent * slope_prior < 0)` AND `|slope_recent - slope_prior| >= median(all movements)`
- **Interpretation**: Volatility is reversing. Order of recent history is critical for trend detection.

#### 7. **high_within_block_dispersion**
- CP-style local average may hide internal structure.
- `dispersion = std(lag1…lag10) / mean(lag1…lag10)`
- Flag: `dispersion >= 80th percentile within asset`
- **Interpretation**: Local vol is volatile. PM's fine-grained phase separation should outperform CP's averaging.

#### 8. **recent_spike_position**
- Spike (max lagged value) is recent.
- Flag: `argmax(lag1…lag10) ∈ {0, 1, 2}` (i.e., lag1, lag2, or lag3)
- **Interpretation**: Recent spike is fresh. PM's order-sensitive features capture recency; CP's averaging dilutes it.

#### 9. **older_spike_position**
- Spike is stale (inside local window but not recent).
- Flag: `argmax(lag1…lag10) ∈ {7, 8, 9}` (i.e., lag8, lag9, or lag10)
- **Interpretation**: Old spike is fading. CP's smoother treatment may be more robust.

#### 10. **same_average_different_path**
- CP's local average is unremarkable, but the temporal path is extreme.
- `local_mean = mean(lag1…lag10)`
- `path_score = |recent_mean_1_5 - older_mean_6_10|`
- Flag: `(local_mean ∈ [40th, 60th percentile]) AND (path_score >= 80th percentile)`
- **Interpretation**: The *structure* of the window dominates, not the level. Pure order-sensitivity test.

## Metrics Computed

### Per-Asset Metrics
- **n_obs**: Observations in this condition for this asset.
- **share_of_asset_obs**: Fraction of the asset's data in this condition.
- **PM_mean_SMAPE / CP_mean_SMAPE**: Average SMAPE% for each model.
- **mean_PM_advantage_vs_CP**: Mean difference in SMAPE% (CP − PM).
- **median_PM_advantage_vs_CP**: Median difference (more robust to outliers).
- **PM_win_rate_vs_CP**: Fraction of observations where PM's error is lower.
- **mean_AbsErr_PM / CP**: Average absolute error for each model.
- **mean_AbsErr_advantage_vs_CP**: Mean difference in MAE (CP − PM).
- **ttest_pval**: Paired t-test p-value on SMAPE loss differential (if scipy available).
- **bootstrap_ci_low / bootstrap_ci_high**: 95% CI for mean PM advantage (1,000 resamples, seed=42).

### Pooled Metrics (Cross-Asset Aggregation)
- **total_n_obs**: Sum of observations across all assets in this condition.
- **number_of_assets**: How many assets contribute to this condition.
- **equal_weight_asset_mean_advantage**: Mean of asset-level mean advantages (each asset counts equally).
- **equal_weight_asset_median_advantage**: Mean of asset-level median advantages.
- **equal_weight_asset_PM_win_rate**: Mean of asset-level win rates.
- **pooled_PM_mean_SMAPE / pooled_CP_mean_SMAPE**: Observation-weighted SMAPE averages.
- **pooled_mean_PM_advantage_vs_CP**: Observation-weighted PM advantage.
- **number_of_assets_where_PM_beats_CP**: Count of assets with positive mean advantage.
- **asset_win_rate**: Fraction of assets where PM's mean advantage is positive.

## Output Files

```
outputs/conditional_pm_cp_tests/
├── README.md (this file)
├── manifest.json                           # Execution metadata
├── scripts/
│   └── run_conditional_pm_cp_tests.py     # Standalone test script
├── results/
│   ├── conditional_summary_by_asset.csv   # Per-asset × condition metrics
│   ├── conditional_summary_pooled.csv     # Pooled (cross-asset) metrics
│   ├── condition_counts.csv                # Condition prevalence by asset
│   └── top_pm_conditions.csv              # Conditions ranked by PM advantage
├── figures/
│   ├── pm_advantage_by_condition.png      # Bar chart: mean PM advantage
│   └── pm_win_rate_by_condition.png       # Bar chart: PM win rate
└── latex/
    └── conditional_summary_pooled.tex     # LaTeX table for paper
```

## Key Findings

### Conditions Where PM Outperforms CP (Equal-Weight Asset Mean)

1. **cluster_entry_strict** (+0.248 SMAPE%)
   - When volatility rises sharply above median.
   - 70% of assets show PM advantage.
   - **Interpretation**: PM's order-aware phase structure captures rising regimes better.

2. **cluster_entry_loose** (+0.248 SMAPE%)
   - When recent volatility exceeds older.
   - 70% of assets show PM advantage.
   - **Interpretation**: Entry signals are PM's strength.

3. **recent_spike_position** (+0.100 SMAPE%)
   - When the highest recent lag is in positions 1–3.
   - 70% of assets show PM advantage.
   - **Interpretation**: PM benefits from recency bias in ordering.

### Conditions Where CP Dominates (or PM Struggles)

1. **high_within_block_dispersion** (−1.825 SMAPE%)
   - When local volatility is highly dispersed.
   - Only 10% of assets show PM advantage.
   - **Interpretation**: High dispersion may confuse PM's phase structure; CP's averaging is more stable.

2. **cluster_exit_strict / cluster_exit_loose** (−1.578 / −1.577 SMAPE%)
   - When volatility falls sharply.
   - Only 10% of assets show PM advantage.
   - **Interpretation**: Exit signals are not PM-friendly; CP's smoothing helps.

3. **same_average_different_path** (−1.512 SMAPE%)
   - When path structure dominates (order matters) but average is mid-range.
   - 0% of assets show PM advantage.
   - **Interpretation**: Pure order sensitivity does not systematically favor PM in this regime.

4. **older_spike_position** (−0.929 SMAPE%)
   - When the max lag is in positions 8–10 (stale).
   - 30% of assets show PM advantage.
   - **Interpretation**: Stale spikes may mislead PM's order-sensitive features.

### Overall (all_observations)

- **Mean PM advantage**: −0.523 SMAPE%
- **PM win rate**: 40.7%
- **Asset win rate**: 20% (2 out of 10 assets show positive advantage on average)
- **Interpretation**: Across the board, CP slightly outperforms PM in average SMAPE%. However, PM shines in specific regimes (entry, recency).

## Limitations

1. **Sample size per asset**: 36,739–36,817 observations (high resolution in time-series context, but heterogeneous across assets).
2. **Feature engineering**: Lagged RV only; does not include volume, bid-ask spread, or order flow intensity.
3. **Model selection**: Expanding-window OLS for both PM and CP; assumes linear relationships.
4. **Statistical power**: Some conditions (e.g., `same_average_different_path`) are rare (~7,400 obs pooled); bootstrapped CIs may be wider.
5. **Correlation**: DM tests assume independence, but time-series forecasts are autocorrelated. Results treated as descriptive.
6. **Cross-asset bias**: Equal-weight aggregation may overweight small-sample assets; observation-weighted pooling is preferred for inference.

## Interpretation Guide

### Green Bars (Positive PM Advantage)
- PM's forecast SMAPE% is lower (better) than CP's.
- PM's order-sensitive phases outperform CP's contiguous blocks.

### Red Bars (Negative PM Advantage)
- CP outperforms PM; PM's order sensitivity becomes a liability.

### Win Rate
- > 50%: PM wins more observations than CP.
- < 50%: CP wins more observations.
- Asset win rate: Fraction of assets where PM's mean SMAPE% is lower.

## Conclusion

**PM vs. CP is context-dependent:**

- **PM excels** during volatility cluster entries and when recent spikes are fresh (lag1–3).
- **CP dominates** during exits, high dispersion, and stale information (lag8–10).
- **Overall**, CP has a slight edge in unconditional performance (−0.52 SMAPE%), but PM's strengths are concentrated in order-sensitive regimes.

**Recommendation**: Consider hybrid or conditional model selection based on market regime. PM's value proposition is strongest during trend reversals and clustering transitions, not in baseline forecasting.

---

**Generated**: See `manifest.json` for timestamp and environment details.  
**Script**: `scripts/run_conditional_pm_cp_tests.py`  
**Data Source**: `run_results/current_intraday/predictions/`  
**Anti-lookahead**: All conditions use `Actual.shift(k)` only.
