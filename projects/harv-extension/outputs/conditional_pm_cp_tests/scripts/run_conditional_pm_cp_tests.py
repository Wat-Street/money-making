"""
Standalone conditional PM-vs-CP test using existing predictions.

This script loads timestamp-level predictions from run_results/current_intraday/predictions/
and tests whether PM's temporal granularity adds value exactly when temporal order should matter most.

Anti-lookahead rule: All condition flags are constructed using lagged actual RV values only.
"""

import os
import sys
import json
import warnings
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent.parent
PRED_DIR = REPO_ROOT / 'run_results' / 'current_intraday' / 'predictions'
OUTPUT_DIR = Path(__file__).parent.parent
RESULTS_DIR = OUTPUT_DIR / 'results'
FIGURES_DIR = OUTPUT_DIR / 'figures'
LATEX_DIR = OUTPUT_DIR / 'latex'

def load_predictions():
    """Load all ticker prediction CSVs."""
    predictions = {}
    if not PRED_DIR.exists():
        raise FileNotFoundError(f"Predictions directory not found: {PRED_DIR}")
    
    for csv_file in sorted(PRED_DIR.glob('*.csv')):
        ticker = csv_file.stem
        try:
            df = pd.read_csv(csv_file, parse_dates=['Date'])
            if 'Actual' not in df.columns or 'Predicted_PM' not in df.columns:
                print(f"Skipping {ticker}: missing required columns")
                continue
            if 'SMAPE_PM_pct' not in df.columns:
                # Compute SMAPE from predictions if missing
                df['SMAPE_PM_pct'] = 200.0 * np.abs(df['Actual'] - df['Predicted_PM']) / \
                                      (np.abs(df['Actual']) + np.abs(df['Predicted_PM']) + 1e-12)
            if 'SMAPE_CP_pct' not in df.columns:
                df['SMAPE_CP_pct'] = 200.0 * np.abs(df['Actual'] - df['Predicted_CP']) / \
                                      (np.abs(df['Actual']) + np.abs(df['Predicted_CP']) + 1e-12)
            if 'AbsErr_PM' not in df.columns:
                df['AbsErr_PM'] = np.abs(df['Actual'] - df['Predicted_PM'])
            if 'AbsErr_CP' not in df.columns:
                df['AbsErr_CP'] = np.abs(df['Actual'] - df['Predicted_CP'])
            
            df = df.sort_values('Date').reset_index(drop=True)
            predictions[ticker] = df
            print(f"Loaded {ticker}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {ticker}: {e}")
    
    return predictions

def add_lagged_features(df, max_lag=22):
    """Add lagged Actual RV values (lag 1 through max_lag)."""
    for lag in range(1, max_lag + 1):
        df[f'lag{lag}'] = df['Actual'].shift(lag)
    return df.dropna(subset=[f'lag{max_lag}'])

def compute_pm_advantage(df):
    """Compute PM advantage over CP."""
    df['PM_advantage_vs_CP'] = df['SMAPE_CP_pct'] - df['SMAPE_PM_pct']
    df['AbsErr_advantage_vs_CP'] = df['AbsErr_CP'] - df['AbsErr_PM']
    return df

def bootstrap_ci(values, n_resamples=1000, seed=42):
    """Compute 95% CI via bootstrap."""
    rng = np.random.default_rng(seed)
    bootstrap_means = []
    for _ in range(n_resamples):
        resample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.nanmean(resample))
    return np.nanpercentile(bootstrap_means, [2.5, 97.5])

def define_conditions(df, asset):
    """Define 8 condition flags (anti-lookahead)."""
    conditions = {}
    
    # Condition 1: all_observations
    conditions['all_observations'] = np.ones(len(df), dtype=bool)
    
    # Prepare aggregates
    recent_mean_1_5 = df[['lag1', 'lag2', 'lag3', 'lag4', 'lag5']].mean(axis=1)
    older_mean_6_10 = df[['lag6', 'lag7', 'lag8', 'lag9', 'lag10']].mean(axis=1)
    recent_mean_1_3 = df[['lag1', 'lag2', 'lag3']].mean(axis=1)
    prior_mean_4_6 = df[['lag4', 'lag5', 'lag6']].mean(axis=1)
    prior_mean_7_10 = df[['lag7', 'lag8', 'lag9', 'lag10']].mean(axis=1)
    local_mean_1_10 = df[['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7', 'lag8', 'lag9', 'lag10']].mean(axis=1)
    local_std_1_10 = df[['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7', 'lag8', 'lag9', 'lag10']].std(axis=1)
    
    # Condition 2: cluster_entry (loose)
    entry_score = recent_mean_1_5 - older_mean_6_10
    entry_threshold = entry_score.quantile(0.80)
    entry_loose = entry_score >= entry_threshold
    conditions['cluster_entry_loose'] = entry_loose
    
    # Condition 2b: cluster_entry (strict)
    median_lagged_rv = df[['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7', 'lag8', 'lag9', 'lag10']].values.flatten()
    median_lagged_rv = np.nanmedian(median_lagged_rv)
    entry_strict = (recent_mean_1_5 >= median_lagged_rv) & entry_loose
    conditions['cluster_entry_strict'] = entry_strict
    
    # Condition 3: cluster_exit (loose)
    exit_score = older_mean_6_10 - recent_mean_1_5
    exit_threshold = exit_score.quantile(0.80)
    exit_loose = exit_score >= exit_threshold
    conditions['cluster_exit_loose'] = exit_loose
    
    # Condition 3b: cluster_exit (strict)
    exit_strict = (older_mean_6_10 >= median_lagged_rv) & exit_loose
    conditions['cluster_exit_strict'] = exit_strict
    
    # Condition 4: inflection_points
    slope_recent = recent_mean_1_3 - prior_mean_4_6
    slope_prior = prior_mean_4_6 - prior_mean_7_10
    inflection_sign_change = (slope_recent * slope_prior) < 0
    slope_movement = np.abs(slope_recent - slope_prior)
    median_slope_movement = slope_movement.median()
    inflection = inflection_sign_change & (slope_movement >= median_slope_movement)
    conditions['inflection_points'] = inflection
    
    # Condition 5: high_within_block_dispersion
    dispersion_1_10 = local_std_1_10 / (local_mean_1_10 + 1e-12)
    dispersion_threshold = dispersion_1_10.quantile(0.80)
    conditions['high_within_block_dispersion'] = dispersion_1_10 >= dispersion_threshold
    
    # Condition 6: recent_spike_position
    max_lag_idx = df[['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7', 'lag8', 'lag9', 'lag10']].values.argmax(axis=1)
    recent_spike = max_lag_idx <= 2  # lag index 0, 1, 2 = lag1, lag2, lag3
    conditions['recent_spike_position'] = recent_spike
    
    # Condition 7: older_spike_position
    older_spike = max_lag_idx >= 7  # lag index 7, 8, 9 = lag8, lag9, lag10
    conditions['older_spike_position'] = older_spike
    
    # Condition 8: same_average_different_path
    path_score = np.abs(recent_mean_1_5 - older_mean_6_10)
    path_threshold = path_score.quantile(0.80)
    local_mean_low = (local_mean_1_10 >= local_mean_1_10.quantile(0.40)) & (local_mean_1_10 <= local_mean_1_10.quantile(0.60))
    conditions['same_average_different_path'] = local_mean_low & (path_score >= path_threshold)
    
    return conditions

def compute_asset_metrics(df, asset, conditions):
    """Compute asset-level metrics for each condition."""
    asset_metrics = []
    
    for cond_name, cond_mask in conditions.items():
        subset = df[cond_mask].copy()
        if len(subset) == 0:
            continue
        
        n_obs = len(subset)
        share_of_obs = n_obs / len(df)
        
        pm_smape = subset['SMAPE_PM_pct'].mean()
        cp_smape = subset['SMAPE_CP_pct'].mean()
        pm_advantage = subset['PM_advantage_vs_CP'].mean()
        pm_advantage_median = subset['PM_advantage_vs_CP'].median()
        pm_win_rate = (subset['PM_advantage_vs_CP'] > 0).mean()
        
        pm_abserr = subset['AbsErr_PM'].mean()
        cp_abserr = subset['AbsErr_CP'].mean()
        abserr_advantage = subset['AbsErr_advantage_vs_CP'].mean()
        
        # Paired t-test on SMAPE loss differential
        loss_diff = subset['SMAPE_CP_pct'].values - subset['SMAPE_PM_pct'].values
        try:
            tstat, pval = stats.ttest_1samp(loss_diff, 0)
            ttest_pval = pval
        except:
            ttest_pval = np.nan
        
        # Bootstrap CI
        try:
            ci_low, ci_high = bootstrap_ci(subset['PM_advantage_vs_CP'].values, seed=42)
        except:
            ci_low, ci_high = np.nan, np.nan
        
        asset_metrics.append({
            'asset': asset,
            'condition': cond_name,
            'n_obs': n_obs,
            'share_of_asset_obs': share_of_obs,
            'PM_mean_SMAPE': pm_smape,
            'CP_mean_SMAPE': cp_smape,
            'mean_PM_advantage_vs_CP': pm_advantage,
            'median_PM_advantage_vs_CP': pm_advantage_median,
            'PM_win_rate_vs_CP': pm_win_rate,
            'mean_AbsErr_PM': pm_abserr,
            'mean_AbsErr_CP': cp_abserr,
            'mean_AbsErr_advantage_vs_CP': abserr_advantage,
            'ttest_pval': ttest_pval,
            'bootstrap_ci_low': ci_low,
            'bootstrap_ci_high': ci_high,
        })
    
    return asset_metrics

def compute_pooled_metrics(all_metrics_df):
    """Compute pooled (cross-asset) metrics for each condition."""
    conditions = all_metrics_df['condition'].unique()
    pooled_metrics = []
    
    for cond in conditions:
        subset = all_metrics_df[all_metrics_df['condition'] == cond]
        
        total_n_obs = subset['n_obs'].sum()
        n_assets = subset['asset'].nunique()
        
        # Equal-weight asset means (each asset counts equally)
        eq_weight_mean_advantage = subset['mean_PM_advantage_vs_CP'].mean()
        eq_weight_median_advantage = subset['median_PM_advantage_vs_CP'].mean()
        eq_weight_win_rate = subset['PM_win_rate_vs_CP'].mean()
        
        # Pooled metrics (weighted by obs count)
        pooled_smape_pm = (subset['n_obs'] * subset['PM_mean_SMAPE']).sum() / total_n_obs
        pooled_smape_cp = (subset['n_obs'] * subset['CP_mean_SMAPE']).sum() / total_n_obs
        pooled_advantage = pooled_smape_cp - pooled_smape_pm
        
        # Estimate pooled win rate (rough approximation)
        n_assets_pm_beats = (subset['mean_PM_advantage_vs_CP'] > 0).sum()
        asset_win_rate = n_assets_pm_beats / n_assets if n_assets > 0 else 0
        
        pooled_metrics.append({
            'condition': cond,
            'total_n_obs': total_n_obs,
            'number_of_assets': n_assets,
            'equal_weight_asset_mean_advantage': eq_weight_mean_advantage,
            'equal_weight_asset_median_advantage': eq_weight_median_advantage,
            'equal_weight_asset_PM_win_rate': eq_weight_win_rate,
            'pooled_PM_mean_SMAPE': pooled_smape_pm,
            'pooled_CP_mean_SMAPE': pooled_smape_cp,
            'pooled_mean_PM_advantage_vs_CP': pooled_advantage,
            'number_of_assets_where_PM_beats_CP': n_assets_pm_beats,
            'asset_win_rate': asset_win_rate,
        })
    
    return pd.DataFrame(pooled_metrics)

def main():
    print("=" * 80)
    print("CONDITIONAL PM-VS-CP TEST (Timestamp-Level Predictions)")
    print("=" * 80)
    
    # Load predictions
    print(f"\n[1] Loading predictions from: {PRED_DIR}")
    predictions = load_predictions()
    tickers = sorted(predictions.keys())
    print(f"Successfully loaded {len(tickers)} tickers: {', '.join(tickers)}")
    
    # Process each asset
    all_asset_metrics = []
    condition_counts = []
    
    for ticker in tickers:
        print(f"\n[2] Processing {ticker}...")
        df = predictions[ticker].copy()
        
        # Add lagged features
        df = add_lagged_features(df, max_lag=22)
        df = compute_pm_advantage(df)
        
        # Define conditions
        conditions = define_conditions(df, ticker)
        
        # Compute asset metrics
        asset_metrics = compute_asset_metrics(df, ticker, conditions)
        all_asset_metrics.extend(asset_metrics)
        
        # Track condition counts
        for cond_name, cond_mask in conditions.items():
            n_true = cond_mask.sum()
            condition_counts.append({
                'asset': ticker,
                'condition': cond_name,
                'count': n_true,
                'pct_of_asset': 100.0 * n_true / len(df),
            })
    
    # Build DataFrames
    print("\n[3] Aggregating metrics...")
    df_asset_metrics = pd.DataFrame(all_asset_metrics)
    df_condition_counts = pd.DataFrame(condition_counts)
    df_pooled_metrics = compute_pooled_metrics(df_asset_metrics)
    
    # Top conditions (by equal-weight advantage)
    df_top_conditions = df_pooled_metrics.sort_values('equal_weight_asset_mean_advantage', ascending=False).copy()
    
    # Save CSVs
    print(f"\n[4] Saving results to {RESULTS_DIR}...")
    df_asset_metrics.to_csv(RESULTS_DIR / 'conditional_summary_by_asset.csv', index=False)
    df_pooled_metrics.to_csv(RESULTS_DIR / 'conditional_summary_pooled.csv', index=False)
    df_condition_counts.to_csv(RESULTS_DIR / 'condition_counts.csv', index=False)
    df_top_conditions.to_csv(RESULTS_DIR / 'top_pm_conditions.csv', index=False)
    print("  ✓ CSVs saved")
    
    # Generate plots
    print(f"\n[5] Generating plots to {FIGURES_DIR}...")
    
    # Plot 1: PM advantage by condition
    fig, ax = plt.subplots(figsize=(12, 6))
    conditions_list = df_pooled_metrics.sort_values('equal_weight_asset_mean_advantage')['condition'].tolist()
    advantages = df_pooled_metrics.sort_values('equal_weight_asset_mean_advantage')['equal_weight_asset_mean_advantage'].tolist()
    colors = ['green' if x > 0 else 'red' for x in advantages]
    ax.barh(conditions_list, advantages, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Equal-Weight Asset Mean PM Advantage vs CP (SMAPE%)')
    ax.set_title('PM Advantage by Condition\n(Positive = PM beats CP)')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pm_advantage_by_condition.png', dpi=100)
    print("  ✓ Advantage plot saved")
    
    # Plot 2: PM win rate by condition
    fig, ax = plt.subplots(figsize=(12, 6))
    conditions_list = df_pooled_metrics.sort_values('equal_weight_asset_PM_win_rate')['condition'].tolist()
    win_rates = df_pooled_metrics.sort_values('equal_weight_asset_PM_win_rate')['equal_weight_asset_PM_win_rate'].tolist()
    colors = ['green' if x > 0.5 else 'red' for x in win_rates]
    ax.barh(conditions_list, win_rates, color=colors, alpha=0.7)
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=0.8, label='50% (neutral)')
    ax.set_xlabel('Equal-Weight Asset PM Win Rate vs CP')
    ax.set_title('PM Win Rate by Condition\n(>50% = PM wins more often)')
    ax.set_xlim([0, 1])
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pm_win_rate_by_condition.png', dpi=100)
    print("  ✓ Win rate plot saved")
    
    # Generate LaTeX table
    print(f"\n[6] Generating LaTeX table to {LATEX_DIR}...")
    df_pooled_display = df_pooled_metrics[[
        'condition', 'total_n_obs', 'number_of_assets',
        'equal_weight_asset_mean_advantage', 'equal_weight_asset_PM_win_rate',
        'asset_win_rate'
    ]].copy()
    df_pooled_display.columns = [
        'Condition', 'Total Obs', 'N Assets',
        'Mean Adv', 'PM Win Rate', 'Asset Win Rate'
    ]
    df_pooled_display = df_pooled_display.round(4)
    
    latex_code = df_pooled_display.to_latex(index=False)
    with open(LATEX_DIR / 'conditional_summary_pooled.tex', 'w') as f:
        f.write(latex_code)
    print("  ✓ LaTeX table saved")
    
    # Generate manifest
    print(f"\n[7] Generating manifest...")
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'platform': platform.platform(),
        'input_directory': str(PRED_DIR),
        'output_directory': str(OUTPUT_DIR),
        'script_path': __file__,
        'tickers_used': tickers,
        'n_tickers': len(tickers),
        'conditions_tested': df_pooled_metrics['condition'].tolist(),
        'n_conditions': len(df_pooled_metrics),
        'total_rows_processed': len(df_asset_metrics),
        'packages': {
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'matplotlib': plt.matplotlib.__version__,
            'scipy': stats.scipy.__version__ if hasattr(stats, 'scipy') else 'available',
        },
        'anti_lookahead': 'All condition flags constructed using lagged Actual RV values only (Actual.shift(k))',
    }
    with open(OUTPUT_DIR / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print("  ✓ Manifest saved")
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTickers processed: {len(tickers)}")
    print(f"Total observations: {len(df_asset_metrics):,}")
    print(f"Conditions tested: {len(df_pooled_metrics)}")
    
    print("\n" + "-" * 80)
    print("POOLED METRICS (All Conditions):")
    print("-" * 80)
    print(df_pooled_metrics[['condition', 'total_n_obs', 'number_of_assets',
                             'equal_weight_asset_mean_advantage', 'equal_weight_asset_PM_win_rate',
                             'asset_win_rate']].to_string(index=False))
    
    print("\n" + "-" * 80)
    print("TOP CONDITIONS WHERE PM BEATS CP:")
    print("-" * 80)
    top_pm = df_top_conditions[df_top_conditions['equal_weight_asset_mean_advantage'] > 0]
    if len(top_pm) > 0:
        print(top_pm[['condition', 'equal_weight_asset_mean_advantage', 'equal_weight_asset_PM_win_rate',
                      'asset_win_rate']].head(10).to_string(index=False))
    else:
        print("No conditions where PM beats CP on average (equal-weighted).")
    
    print("\n" + "-" * 80)
    print("CONDITIONS WHERE CP DOMINATES:")
    print("-" * 80)
    bottom_pm = df_top_conditions[df_top_conditions['equal_weight_asset_mean_advantage'] < 0]
    if len(bottom_pm) > 0:
        print(bottom_pm[['condition', 'equal_weight_asset_mean_advantage', 'equal_weight_asset_PM_win_rate',
                         'asset_win_rate']].head(10).to_string(index=False))
    else:
        print("No conditions where CP dominates (all show PM advantage or neutral).")
    
    print("\n" + "=" * 80)
    print(f"✓ Test completed successfully!")
    print(f"Output folder: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()
