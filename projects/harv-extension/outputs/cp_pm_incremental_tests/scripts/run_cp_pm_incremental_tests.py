"""
CP + PM Incremental Value Test
================================
Validate recomputed CP against existing saved CP,
then test whether CP + PM beats CP on conditional states.

This script:
1. Loads existing CP predictions
2. Recomputes CP and validates against existing
3. Recomputes CP+PM
4. Compares hybrid performance on conditional states
5. Generates comparison metrics

Anti-lookahead rule: All conditions use Actual.shift(k) for k >= 1.
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
from statsmodels.api import OLS, add_constant

warnings.filterwarnings('ignore')

# Add repo code to path
REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code'))

# Import utilities from repo
from utils.data_utils import (
    fetch_intraday_data, 
    calculate_intraday_realized_volatility, 
    fit_and_predict_extended
)
from utils.models_utils import contig_prime_modulo, add_prime_modulo_terms
from utils.harvey_utils import add_harv_terms

# Paths
PRED_DIR = REPO_ROOT / 'run_results' / 'current_intraday' / 'predictions'
DATA_DIR = REPO_ROOT / 'data' / 'market_data' / 'clean'
OUTPUT_DIR = REPO_ROOT / 'outputs' / 'cp_pm_incremental_tests'
RESULTS_DIR = OUTPUT_DIR / 'results'
FIGURES_DIR = OUTPUT_DIR / 'figures'
LATEX_DIR = OUTPUT_DIR / 'latex'
CP_VALIDATION_WARMUP = 600
CP_VALIDATION_FEATURE_PREFIXES = ('RV', 'CP')
HYBRID_FEATURE_PREFIXES = ('RV', 'CP', 'PM')
SHIFT_OFFSETS = [-2, -1, 0, 1, 2]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR.mkdir(parents=True, exist_ok=True)


def select_feature_columns(frame, prefixes):
    return [c for c in frame.columns if c.startswith(prefixes)]

def load_existing_predictions():
    """Load existing predictions with Predicted_CP."""
    predictions = {}
    if not PRED_DIR.exists():
        raise FileNotFoundError(f"Predictions directory not found: {PRED_DIR}")
    
    for csv_file in sorted(PRED_DIR.glob('*.csv')):
        ticker = csv_file.stem
        try:
            df = pd.read_csv(csv_file, parse_dates=['Date'])
            if 'Actual' not in df.columns or 'Predicted_CP' not in df.columns:
                print(f"⚠ Skipping {ticker}: missing Actual or Predicted_CP")
                continue
            
            df = df.sort_values('Date').reset_index(drop=True)
            predictions[ticker] = df
            print(f"✓ Loaded {ticker}: {len(df)} rows")
        except Exception as e:
            print(f"✗ Error loading {ticker}: {e}")
    
    return predictions

def recompute_cp_forecasts(ticker, data_dir=None):
    """Recompute CP forecasts from raw data."""
    if data_dir is None:
        data_dir = DATA_DIR
    
    # Load raw intraday data
    try:
        raw = fetch_intraday_data(ticker, use_local=True, local_dir=str(data_dir))
    except Exception as e:
        print(f"  ⚠ Could not load raw data for {ticker}: {e}")
        return None
    
    # Calculate RV
    try:
        vol = calculate_intraday_realized_volatility(raw)
    except Exception as e:
        print(f"  ⚠ Could not compute RV for {ticker}: {e}")
        return None
    
    # Apply CP feature engineering
    try:
        extended = contig_prime_modulo(vol.copy(), n=22, per_day_normalize=False)
    except Exception as e:
        print(f"  ⚠ Could not apply CP to {ticker}: {e}")
        return None
    
    # Extract features and predict
    try:
        features = select_feature_columns(extended, CP_VALIDATION_FEATURE_PREFIXES)
        if len(features) == 0:
            print(f"  ⚠ No CP features found for {ticker}")
            return None
        
        preds = fit_and_predict_extended(
            extended, features, n=22, warmup=CP_VALIDATION_WARMUP, model_name='CP'
        )
        
        if preds is None or preds.empty:
            print(f"  ⚠ No predictions for {ticker}")
            return None
        
        return preds
    except Exception as e:
        print(f"  ⚠ Could not fit/predict for {ticker}: {e}")
        return None


def build_cp_alignment_shift_diagnostics(existing_df, recomputed_df, ticker):
    """Check whether a simple row shift explains CP mismatches."""
    diagnostics = []
    existing_vals = existing_df[['Date', 'Predicted_CP']].copy()
    recomputed_vals = recomputed_df.reset_index().rename(columns={'index': 'Date'})[['Date', 'Predicted_CP']].copy()

    for shift in SHIFT_OFFSETS:
        shifted = recomputed_vals.copy()
        shifted['Predicted_CP'] = shifted['Predicted_CP'].shift(shift)
        merged = existing_vals.merge(shifted, on='Date', how='inner', suffixes=('_existing', '_shifted'))
        merged = merged.sort_values('Date').reset_index(drop=True)

        if merged.empty:
            diagnostics.append({
                'ticker': ticker,
                'shift': shift,
                'n_aligned': 0,
                'mean_abs_diff': np.nan,
                'median_abs_diff': np.nan,
                'max_abs_diff': np.nan,
                'correlation': np.nan,
                'first_aligned_date': None,
                'last_aligned_date': None,
            })
            continue

        existing_series = merged['Predicted_CP_existing'].astype(float)
        shifted_series = merged['Predicted_CP_shifted'].astype(float)
        abs_diff = (existing_series - shifted_series).abs()

        diagnostics.append({
            'ticker': ticker,
            'shift': shift,
            'n_aligned': len(merged),
            'mean_abs_diff': float(abs_diff.mean()),
            'median_abs_diff': float(abs_diff.median()),
            'max_abs_diff': float(abs_diff.max()),
            'correlation': float(existing_series.corr(shifted_series)),
            'first_aligned_date': merged['Date'].iloc[0],
            'last_aligned_date': merged['Date'].iloc[-1],
        })

    return pd.DataFrame(diagnostics)

def validate_cp_recomputation(existing_df, recomputed_df):
    """Compare existing Predicted_CP with recomputed CP."""
    # Align on Date index
    try:
        merged = existing_df[['Date', 'Predicted_CP']].merge(
            recomputed_df.reset_index().rename(columns={'index': 'Date'})[['Date', 'Predicted_CP']],
            on='Date',
            suffixes=('_existing', '_recomputed'),
            how='inner'
        )
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'n_aligned': 0,
        }
    
    if merged.empty:
        return {
            'status': 'failed',
            'message': 'No aligned dates between existing and recomputed',
            'n_aligned': 0,
        }
    
    existing_vals = merged['Predicted_CP_existing'].values
    recomputed_vals = merged['Predicted_CP_recomputed'].values
    
    # Compute validation metrics
    abs_diff = np.abs(existing_vals - recomputed_vals)
    mean_abs_diff = np.mean(abs_diff)
    median_abs_diff = np.median(abs_diff)
    max_abs_diff = np.max(abs_diff)
    
    # Correlation
    try:
        corr = np.corrcoef(existing_vals, recomputed_vals)[0, 1]
    except:
        corr = np.nan

    first_aligned_date = merged['Date'].min()
    last_aligned_date = merged['Date'].max()
    
    # Threshold for pass/fail
    passes = corr > 0.95 and mean_abs_diff < 1.0
    
    return {
        'status': 'passed' if passes else 'failed',
        'n_aligned': len(merged),
        'n_existing': len(existing_df),
        'n_recomputed': len(recomputed_df),
        'mean_existing_CP': float(np.mean(existing_vals)),
        'mean_recomputed_CP': float(np.mean(recomputed_vals)),
        'std_existing_CP': float(np.std(existing_vals, ddof=1)) if len(existing_vals) > 1 else np.nan,
        'std_recomputed_CP': float(np.std(recomputed_vals, ddof=1)) if len(recomputed_vals) > 1 else np.nan,
        'mean_abs_diff': mean_abs_diff,
        'median_abs_diff': median_abs_diff,
        'max_abs_diff': max_abs_diff,
        'correlation': corr,
        'first_aligned_date': first_aligned_date,
        'last_aligned_date': last_aligned_date,
        'passes': passes,
    }

def recompute_cp_pm_forecasts(ticker, data_dir=None):
    """Recompute CP+PM (hybrid) forecasts from raw data."""
    if data_dir is None:
        data_dir = DATA_DIR
    
    # Load raw intraday data
    try:
        raw = fetch_intraday_data(ticker, use_local=True, local_dir=str(data_dir))
    except Exception as e:
        return None
    
    # Calculate RV
    try:
        vol = calculate_intraday_realized_volatility(raw)
    except Exception as e:
        return None
    
    # Apply CP + PM feature engineering
    try:
        extended = contig_prime_modulo(vol.copy(), n=22, per_day_normalize=False)
        extended = add_prime_modulo_terms(extended.copy(), n=22)
    except Exception as e:
        return None
    
    # Extract all CP and PM features (not HAR for now)
    try:
        features = select_feature_columns(extended, HYBRID_FEATURE_PREFIXES)
        if len(features) == 0:
            return None
        
        preds = fit_and_predict_extended(
            extended, features, n=22, warmup=CP_VALIDATION_WARMUP, model_name='CP_PLUS_PM'
        )
        
        if preds is None or preds.empty:
            return None
        
        return preds
    except Exception as e:
        return None

def add_lagged_features(df, max_lag=22):
    """Add lagged Actual RV values (anti-lookahead)."""
    for lag in range(1, max_lag + 1):
        df[f'lag{lag}'] = df['Actual'].shift(lag)
    return df.dropna(subset=[f'lag{max_lag}'])

def define_conditions(df):
    """Define 10 condition flags (anti-lookahead)."""
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
    
    # Condition 2 & 3: cluster entry
    entry_score = recent_mean_1_5 - older_mean_6_10
    entry_threshold = entry_score.quantile(0.80)
    entry_loose = entry_score >= entry_threshold
    conditions['cluster_entry_loose'] = entry_loose
    
    median_lagged_rv = df[['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7', 'lag8', 'lag9', 'lag10']].values.flatten()
    median_lagged_rv = np.nanmedian(median_lagged_rv)
    entry_strict = (recent_mean_1_5 >= median_lagged_rv) & entry_loose
    conditions['cluster_entry_strict'] = entry_strict
    
    # Condition 4 & 5: cluster exit
    exit_score = older_mean_6_10 - recent_mean_1_5
    exit_threshold = exit_score.quantile(0.80)
    exit_loose = exit_score >= exit_threshold
    conditions['cluster_exit_loose'] = exit_loose
    
    exit_strict = (older_mean_6_10 >= median_lagged_rv) & exit_loose
    conditions['cluster_exit_strict'] = exit_strict
    
    # Condition 6: inflection points
    slope_recent = recent_mean_1_3 - prior_mean_4_6
    slope_prior = prior_mean_4_6 - prior_mean_7_10
    inflection_sign_change = (slope_recent * slope_prior) < 0
    slope_movement = np.abs(slope_recent - slope_prior)
    median_slope_movement = slope_movement.median()
    inflection = inflection_sign_change & (slope_movement >= median_slope_movement)
    conditions['inflection_points'] = inflection
    
    # Condition 7: high within-block dispersion
    dispersion_1_10 = local_std_1_10 / (local_mean_1_10 + 1e-12)
    dispersion_threshold = dispersion_1_10.quantile(0.80)
    conditions['high_within_block_dispersion'] = dispersion_1_10 >= dispersion_threshold
    
    # Condition 8: recent spike
    max_lag_idx = df[['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7', 'lag8', 'lag9', 'lag10']].values.argmax(axis=1)
    recent_spike = max_lag_idx <= 2
    conditions['recent_spike_position'] = recent_spike
    
    # Condition 9: older spike
    older_spike = max_lag_idx >= 7
    conditions['older_spike_position'] = older_spike
    
    # Condition 10: same average, different path
    path_score = np.abs(recent_mean_1_5 - older_mean_6_10)
    path_threshold = path_score.quantile(0.80)
    local_mean_low = (local_mean_1_10 >= local_mean_1_10.quantile(0.40)) & (local_mean_1_10 <= local_mean_1_10.quantile(0.60))
    conditions['same_average_different_path'] = local_mean_low & (path_score >= path_threshold)
    
    return conditions

def bootstrap_ci(values, n_resamples=1000, seed=42):
    """Compute 95% CI via bootstrap."""
    rng = np.random.default_rng(seed)
    bootstrap_means = []
    for _ in range(n_resamples):
        resample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.nanmean(resample))
    return np.nanpercentile(bootstrap_means, [2.5, 97.5])

def compute_asset_metrics(df, asset, conditions):
    """Compute asset-level metrics for each condition."""
    asset_metrics = []
    
    for cond_name, cond_mask in conditions.items():
        subset = df[cond_mask].copy()
        if len(subset) == 0:
            continue
        
        n_obs = len(subset)
        
        # CP vs Hybrid comparison
        cp_smape = subset['SMAPE_CP_pct'].mean()
        hybrid_smape = subset['SMAPE_Hybrid_pct'].mean()
        hybrid_advantage = cp_smape - hybrid_smape  # positive = hybrid beats CP
        hybrid_advantage_median = (subset['SMAPE_CP_pct'] - subset['SMAPE_Hybrid_pct']).median()
        hybrid_win_rate = (subset['SMAPE_CP_pct'] > subset['SMAPE_Hybrid_pct']).mean()
        
        cp_abserr = subset['AbsErr_CP'].mean()
        hybrid_abserr = subset['AbsErr_Hybrid'].mean()
        abserr_advantage = cp_abserr - hybrid_abserr
        
        # Paired t-test
        loss_diff = subset['SMAPE_CP_pct'].values - subset['SMAPE_Hybrid_pct'].values
        try:
            tstat, pval = stats.ttest_1samp(loss_diff, 0)
            ttest_pval = pval
        except:
            ttest_pval = np.nan
        
        # Bootstrap CI
        try:
            ci_low, ci_high = bootstrap_ci(loss_diff, seed=42)
        except:
            ci_low, ci_high = np.nan, np.nan
        
        asset_metrics.append({
            'asset': asset,
            'condition': cond_name,
            'n_obs': n_obs,
            'CP_mean_SMAPE': cp_smape,
            'Hybrid_mean_SMAPE': hybrid_smape,
            'mean_hybrid_advantage_vs_CP': hybrid_advantage,
            'median_hybrid_advantage_vs_CP': hybrid_advantage_median,
            'hybrid_win_rate_vs_CP': hybrid_win_rate,
            'mean_AbsErr_CP': cp_abserr,
            'mean_AbsErr_Hybrid': hybrid_abserr,
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
        
        # Equal-weight asset means
        eq_weight_mean_advantage = subset['mean_hybrid_advantage_vs_CP'].mean()
        eq_weight_median_advantage = subset['median_hybrid_advantage_vs_CP'].mean()
        eq_weight_win_rate = subset['hybrid_win_rate_vs_CP'].mean()
        
        # Pooled metrics (weighted by obs)
        pooled_smape_cp = (subset['n_obs'] * subset['CP_mean_SMAPE']).sum() / total_n_obs
        pooled_smape_hybrid = (subset['n_obs'] * subset['Hybrid_mean_SMAPE']).sum() / total_n_obs
        pooled_advantage = pooled_smape_cp - pooled_smape_hybrid
        
        n_assets_hybrid_beats = (subset['mean_hybrid_advantage_vs_CP'] > 0).sum()
        asset_win_rate = n_assets_hybrid_beats / n_assets if n_assets > 0 else 0
        
        pooled_metrics.append({
            'condition': cond,
            'total_n_obs': total_n_obs,
            'number_of_assets': n_assets,
            'equal_weight_asset_mean_advantage': eq_weight_mean_advantage,
            'equal_weight_asset_median_advantage': eq_weight_median_advantage,
            'equal_weight_asset_hybrid_win_rate': eq_weight_win_rate,
            'pooled_CP_mean_SMAPE': pooled_smape_cp,
            'pooled_Hybrid_mean_SMAPE': pooled_smape_hybrid,
            'pooled_mean_hybrid_advantage_vs_CP': pooled_advantage,
            'number_of_assets_where_hybrid_beats_CP': n_assets_hybrid_beats,
            'asset_win_rate': asset_win_rate,
        })
    
    return pd.DataFrame(pooled_metrics)

def main():
    print("=" * 90)
    print("CP + PM INCREMENTAL VALUE TEST")
    print("=" * 90)
    
    # Step 1: Load existing predictions
    print("\n[1] Loading existing predictions...")
    existing_predictions = load_existing_predictions()
    tickers = sorted(existing_predictions.keys())
    print(f"✓ Loaded {len(tickers)} tickers: {', '.join(tickers)}")
    
    # Step 2: Validate CP recomputation
    print("\n[2] Validating CP recomputation...")
    validation_results = []
    shift_diagnostics = []
    valid_tickers = []
    
    for ticker in tickers:
        print(f"  Recomputing CP for {ticker}...")
        recomputed_cp = recompute_cp_forecasts(ticker)
        
        if recomputed_cp is None:
            print(f"  ✗ Failed to recompute CP for {ticker}")
            validation_results.append({
                'ticker': ticker,
                'status': 'failed',
                'n_aligned': 0,
                'n_existing': len(existing_predictions[ticker]),
                'n_recomputed': 0,
                'mean_abs_diff': np.nan,
                'median_abs_diff': np.nan,
                'max_abs_diff': np.nan,
                'correlation': np.nan,
                'mean_existing_CP': np.nan,
                'mean_recomputed_CP': np.nan,
                'std_existing_CP': np.nan,
                'std_recomputed_CP': np.nan,
                'first_aligned_date': None,
                'last_aligned_date': None,
                'passes': False,
            })
            continue
        
        existing_df = existing_predictions[ticker].copy()
        val_result = validate_cp_recomputation(existing_df, recomputed_cp)
        val_result['ticker'] = ticker
        validation_results.append(val_result)
        shift_diagnostics.append(build_cp_alignment_shift_diagnostics(existing_df, recomputed_cp, ticker))
        
        if val_result.get('passes', False):
            valid_tickers.append(ticker)
            print(f"  ✓ CP validation passed: corr={val_result['correlation']:.4f}, mean_diff={val_result['mean_abs_diff']:.6f}")
        else:
            print(f"  ✗ CP validation FAILED: corr={val_result.get('correlation', np.nan):.4f}")
    
    df_validation = pd.DataFrame(validation_results)
    df_validation.to_csv(RESULTS_DIR / 'cp_recompute_validation.csv', index=False)
    df_shift_diagnostics = pd.concat(shift_diagnostics, ignore_index=True) if shift_diagnostics else pd.DataFrame()
    df_shift_diagnostics.to_csv(RESULTS_DIR / 'cp_alignment_shift_diagnostics.csv', index=False)
    print(f"\n✓ Validation results saved")
    
    # Check if enough tickers passed validation
    if len(valid_tickers) < 3:
        print(f"\n✗ CRITICAL: Only {len(valid_tickers)} tickers passed validation. Stopping.")
        with open(OUTPUT_DIR / 'README.md', 'w') as f:
            f.write("# CP + PM Incremental Test - VALIDATION FAILED\n\n")
            f.write(f"Only {len(valid_tickers)} out of {len(tickers)} tickers passed CP recomputation validation.\n\n")
            f.write("## Validation Results\n\n")
            f.write(df_validation.to_string())
            if not df_shift_diagnostics.empty:
                f.write("\n\n## Shift Diagnostics\n\n")
                f.write(df_shift_diagnostics.to_string())
        return
    
    print(f"✓ {len(valid_tickers)}/{len(tickers)} tickers passed validation. Proceeding.")
    tickers = valid_tickers
    
    # Step 3: Recompute CP+PM and compare
    print("\n[3] Recomputing CP+PM forecasts and comparing...")
    all_asset_metrics = []
    
    for ticker in tickers:
        print(f"  Processing {ticker}...")
        
        # Get existing CP
        existing_df = existing_predictions[ticker].copy()
        
        # Recompute Hybrid (CP+PM)
        hybrid_df = recompute_cp_pm_forecasts(ticker)
        if hybrid_df is None or hybrid_df.empty:
            print(f"    ⚠ Could not recompute hybrid for {ticker}")
            continue
        
        # Merge: existing CP with recomputed hybrid
        try:
            merged = existing_df[['Date', 'Actual', 'Predicted_CP']].merge(
                hybrid_df.reset_index().rename(columns={'index': 'Date'})[['Date', 'Predicted_CP_PLUS_PM']],
                on='Date',
                how='inner'
            )
        except:
            print(f"    ⚠ Could not merge forecasts for {ticker}")
            continue
        
        if merged.empty:
            print(f"    ⚠ No aligned rows for {ticker}")
            continue
        
        merged = merged.rename(columns={'Predicted_CP_PLUS_PM': 'Predicted_Hybrid'})
        
        # Compute errors
        merged['AbsErr_CP'] = np.abs(merged['Actual'] - merged['Predicted_CP'])
        merged['AbsErr_Hybrid'] = np.abs(merged['Actual'] - merged['Predicted_Hybrid'])
        merged['SMAPE_CP_pct'] = 200.0 * merged['AbsErr_CP'] / (np.abs(merged['Actual']) + np.abs(merged['Predicted_CP']) + 1e-12)
        merged['SMAPE_Hybrid_pct'] = 200.0 * merged['AbsErr_Hybrid'] / (np.abs(merged['Actual']) + np.abs(merged['Predicted_Hybrid']) + 1e-12)
        
        # Add lagged features and define conditions
        merged = add_lagged_features(merged, max_lag=22)
        conditions = define_conditions(merged)
        
        # Compute metrics
        asset_metrics = compute_asset_metrics(merged, ticker, conditions)
        all_asset_metrics.extend(asset_metrics)
    
    if len(all_asset_metrics) == 0:
        print("✗ No valid metrics computed. Stopping.")
        return
    
    # Step 4: Aggregate and save results
    print("\n[4] Aggregating results...")
    df_asset_metrics = pd.DataFrame(all_asset_metrics)
    df_pooled_metrics = compute_pooled_metrics(df_asset_metrics)
    df_top_conditions = df_pooled_metrics.sort_values('equal_weight_asset_mean_advantage', ascending=False).copy()
    
    df_asset_metrics.to_csv(RESULTS_DIR / 'fresh_conditional_hybrid_summary_by_asset.csv', index=False)
    df_pooled_metrics.to_csv(RESULTS_DIR / 'fresh_conditional_hybrid_summary_pooled.csv', index=False)
    df_top_conditions.to_csv(RESULTS_DIR / 'top_fresh_hybrid_conditions.csv', index=False)
    
    # Hybrid predictions summary
    hybrid_summary = df_pooled_metrics[['condition', 'pooled_CP_mean_SMAPE', 'pooled_Hybrid_mean_SMAPE', 'pooled_mean_hybrid_advantage_vs_CP']].copy()
    hybrid_summary.to_csv(RESULTS_DIR / 'fresh_cp_vs_hybrid_predictions_summary.csv', index=False)
    
    print(f"✓ Results saved")
    
    # Step 5: Generate plots
    print("\n[5] Generating visualizations...")
    
    # Plot 1: Hybrid advantage by condition
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_df = df_pooled_metrics.sort_values('equal_weight_asset_mean_advantage')
    conditions_list = sorted_df['condition'].tolist()
    advantages = sorted_df['equal_weight_asset_mean_advantage'].tolist()
    colors = ['green' if x > 0 else 'red' for x in advantages]
    ax.barh(conditions_list, advantages, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Equal-Weight Asset Mean Hybrid Advantage vs CP (SMAPE%)')
    ax.set_title('Hybrid (CP+PM) Advantage by Condition\n(Positive = CP+PM beats CP)')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fresh_hybrid_advantage_by_condition.png', dpi=100)
    
    # Plot 2: Hybrid win rate by condition
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_df = df_pooled_metrics.sort_values('equal_weight_asset_hybrid_win_rate')
    conditions_list = sorted_df['condition'].tolist()
    win_rates = sorted_df['equal_weight_asset_hybrid_win_rate'].tolist()
    colors = ['green' if x > 0.5 else 'red' for x in win_rates]
    ax.barh(conditions_list, win_rates, color=colors, alpha=0.7)
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=0.8, label='50% (neutral)')
    ax.set_xlabel('Equal-Weight Asset Hybrid Win Rate vs CP')
    ax.set_title('Hybrid (CP+PM) Win Rate by Condition\n(>50% = Hybrid wins more often)')
    ax.set_xlim([0, 1])
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fresh_hybrid_win_rate_by_condition.png', dpi=100)
    
    print(f"✓ Plots saved")
    
    # Step 6: Generate LaTeX table
    print("\n[6] Generating LaTeX table...")
    df_latex = df_pooled_metrics[[
        'condition', 'total_n_obs', 'number_of_assets',
        'equal_weight_asset_mean_advantage', 'equal_weight_asset_hybrid_win_rate',
        'asset_win_rate'
    ]].copy()
    df_latex.columns = [
        'Condition', 'Total Obs', 'N Assets',
        'Mean Adv', 'Hybrid Win Rate', 'Asset Win Rate'
    ]
    df_latex = df_latex.round(4)
    latex_code = df_latex.to_latex(index=False)
    with open(LATEX_DIR / 'fresh_conditional_hybrid_summary_pooled.tex', 'w') as f:
        f.write(latex_code)
    print(f"✓ LaTeX table saved")
    
    # Step 7: Generate manifest
    print("\n[7] Generating manifest...")
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'platform': platform.platform(),
        'input_directory': str(PRED_DIR),
        'data_directory': str(DATA_DIR),
        'output_directory': str(OUTPUT_DIR),
        'script_path': __file__,
        'tickers_used': tickers,
        'n_tickers': len(tickers),
        'validation_passed': True,
        'cp_validation_warmup': CP_VALIDATION_WARMUP,
        'cp_validation_feature_prefixes': list(CP_VALIDATION_FEATURE_PREFIXES),
        'hybrid_feature_prefixes': list(HYBRID_FEATURE_PREFIXES),
        'conditions_tested': df_pooled_metrics['condition'].tolist(),
        'n_conditions': len(df_pooled_metrics),
        'anti_lookahead': 'All condition flags constructed using lagged Actual RV values only',
    }
    with open(OUTPUT_DIR / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Step 8: Generate README
    print("\n[8] Generating README...")
    with open(OUTPUT_DIR / 'README.md', 'w') as f:
        f.write("# CP + PM Incremental Value Test\n\n")
        f.write(f"**Status**: ✅ SUCCESS\n\n")
        f.write("Saved CP was reproduced by mirroring the original benchmark's warmup and feature selection; the incremental test therefore uses the saved CP baseline.\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Tickers tested**: {len(tickers)} ({', '.join(tickers)})\n")
        f.write(f"- **Total observations analyzed**: {len(df_asset_metrics):,}\n")
        f.write(f"- **Conditions tested**: {len(df_pooled_metrics)}\n")
        f.write(f"- **CP validation**: PASSED ({len(valid_tickers)}/{len(existing_predictions)} tickers)\n\n")
        f.write("## Key Findings\n\n")
        f.write("### Hybrid (CP+PM) vs CP Performance\n\n")
        f.write(df_pooled_metrics[['condition', 'equal_weight_asset_mean_advantage', 'equal_weight_asset_hybrid_win_rate', 'asset_win_rate']].to_markdown())
        f.write("\n\n## Output Files\n\n")
        f.write("- `cp_recompute_validation.csv` - CP recomputation validation results\n")
        f.write("- `cp_alignment_shift_diagnostics.csv` - Shift diagnostics for recomputed CP\n")
        f.write("- `fresh_conditional_hybrid_summary_by_asset.csv` - Per-asset × condition metrics\n")
        f.write("- `fresh_conditional_hybrid_summary_pooled.csv` - Pooled (cross-asset) metrics\n")
        f.write("- `fresh_cp_vs_hybrid_predictions_summary.csv` - Summary of SMAPE comparisons\n")
        f.write("- `top_fresh_hybrid_conditions.csv` - Conditions ranked by advantage\n")
        f.write("- `manifest.json` - Execution metadata\n")
        f.write("- `figures/fresh_hybrid_advantage_by_condition.png` - Bar chart of advantages\n")
        f.write("- `figures/fresh_hybrid_win_rate_by_condition.png` - Bar chart of win rates\n")
        f.write("- `latex/fresh_conditional_hybrid_summary_pooled.tex` - LaTeX table for paper\n")
    
    print(f"✓ README generated")
    
    # Final summary
    print("\n" + "=" * 90)
    print("✅ TEST COMPLETED SUCCESSFULLY")
    print("=" * 90)
    print(f"\nPooled Results (All Conditions):")
    print(df_pooled_metrics[['condition', 'equal_weight_asset_mean_advantage', 'equal_weight_asset_hybrid_win_rate']].to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("=" * 90)

if __name__ == '__main__':
    main()
