import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from .metrics_utils import compute_metrics, dm_test, fisher_p, per_timestamp_smape_percent

def _se(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return float('nan')
    return float(np.nanstd(x, ddof=1) / np.sqrt(x.size))

# ---------------------------
# Section A (Overall accuracy)
# ---------------------------

def aggregate_overall_from_perasset(per_asset_rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(per_asset_rows)
    if df.empty:
        return pd.DataFrame(columns=[
            'model','smape_pct_mean','smape_pct_se','rmse_mean','rmse_se',
            'mae_mean','mae_se','medae_mean','medae_se',
            'diracc_pct_mean','diracc_pct_se','n_test_mean','sig_mark_smape'
        ])
    models = sorted(df['model'].unique(), key=lambda x: x)
    rows = []
    base_means = None
    if 'HAR' in models:
        base_means = dict(
            df[df['model'] == 'HAR']
            .groupby('model')[['smape_pct','rmse','mae','diracc_pct']]
            .mean().iloc[0]
        )
    for m in models:
        sub = df[df['model'] == m]
        means = sub[['smape_pct','rmse','mae','medae','diracc_pct','n_test']].mean()
        ses = sub[['smape_pct','rmse','mae','medae','diracc_pct']].apply(lambda s: _se(s.values))
        mark = ''
        if m != 'HAR' and 'dm_p_mae' in sub.columns and base_means is not None:
            pval = fisher_p(sub['dm_p_mae'].values)
            # IMPORTANT: only celebrate significance if the SMAPE mean improved vs HAR
            if np.isfinite(pval) and means['smape_pct'] < base_means['smape_pct']:
                if pval < 1e-3:
                    mark = '★'
                elif pval < 1e-2:
                    mark = '‡'
                elif pval < 5e-2:
                    mark = '†'
        rows.append({
            'model': m,
            'smape_pct_mean': float(means['smape_pct']),
            'smape_pct_se': float(ses['smape_pct']),
            'rmse_mean': float(means['rmse']),
            'rmse_se': float(ses['rmse']),
            'mae_mean': float(means['mae']),
            'mae_se': float(ses['mae']),
            'medae_mean': float(means['medae']),
            'medae_se': float(ses['medae']),
            'diracc_pct_mean': float(means['diracc_pct']),
            'diracc_pct_se': float(ses['diracc_pct']),
            'n_test_mean': float(means['n_test']),
            'sig_mark_smape': mark,
        })
    return pd.DataFrame(rows).sort_values('model')

def aggregate_overall_from_predictions(pred_path: str, models: List[str]) -> Tuple[pd.DataFrame, List[dict]]:
    per_asset_rows = []
    if not os.path.isdir(pred_path):
        return aggregate_overall_from_perasset([]), []
    for csv in sorted(os.listdir(pred_path)):
        if not csv.endswith('.csv'):
            continue
        asset = os.path.splitext(csv)[0]
        df = pd.read_csv(os.path.join(pred_path, csv), parse_dates=['Date']).set_index('Date')
        if 'Actual' not in df.columns:
            continue
        available = [m for m in models if f'Predicted_{m}' in df.columns]
        base_abs = None
        if 'HAR' in available:
            base_abs = np.abs((df['Actual'] - df['Predicted_HAR']).values)
        for m in available:
            y = df['Actual'].values.astype(float)
            yhat = df[f'Predicted_{m}'].values.astype(float)
            mask = np.isfinite(y) & np.isfinite(yhat)
            y = y[mask]; yhat = yhat[mask]
            if y.size < 10:
                continue
            mets = compute_metrics(y, yhat)
            dm_p = np.nan
            if base_abs is not None:
                loss_a = base_abs[mask]
                loss_b = np.abs(y - yhat)
                _, dm_p = dm_test(loss_a, loss_b)
            per_asset_rows.append({
                'asset': asset,
                'model': m,
                'smape_pct': mets['smape_pct'],
                'rmse': mets['rmse'],
                'mae': mets['mae'],
                'medae': mets['medae'],
                'diracc_pct': mets['diracc_pct'],
                'n_test': mets['n_test'],
                'dm_p_mae': dm_p,
            })
    overall = aggregate_overall_from_perasset(per_asset_rows)
    return overall, per_asset_rows


def save_table_overall(df: pd.DataFrame, outfile_csv: str):
    os.makedirs(os.path.dirname(outfile_csv), exist_ok=True)
    df.to_csv(outfile_csv, index=False)

def select_best_by_smape(overall_df: pd.DataFrame, candidates: List[str]) -> str:
    sub = overall_df[overall_df['model'].isin(candidates)].copy()
    if sub.empty:
        return None
    return sub.sort_values('smape_pct_mean').iloc[0]['model']

# ---------------------------
# Section B (Regime analysis)
# ---------------------------

def _bucket_masks_by_quantiles(y: np.ndarray, qlow: float = 0.33, qhigh: float = 0.66):
    """Return boolean masks for low, med, high based on y quantiles."""
    y = np.asarray(y, dtype=float)
    q1 = np.nanquantile(y, qlow)
    q2 = np.nanquantile(y, qhigh)
    low = y <= q1
    med = (y > q1) & (y <= q2)
    high = y > q2
    return low, med, high

def aggregate_regimes_from_predictions(pred_path: str, models: List[str], qlow: float = 0.33, qhigh: float = 0.66) -> pd.DataFrame:
    """
    For each asset, bucket timestamps by Actual's quantiles into Low/Med/High.
    Compute SMAPE% within each bucket for each model.
    Aggregate across assets: mean ± s.e.
    """
    per_asset_bucket = []  # one row per (asset, model) with low/med/high smape
    if not os.path.isdir(pred_path):
        return pd.DataFrame(columns=[
            'model','smape_low_pct_mean','smape_low_pct_se',
            'smape_med_pct_mean','smape_med_pct_se',
            'smape_high_pct_mean','smape_high_pct_se'
        ])

    for csv in sorted(os.listdir(pred_path)):
        if not csv.endswith('.csv'):
            continue
        asset = os.path.splitext(csv)[0]
        df = pd.read_csv(os.path.join(pred_path, csv), parse_dates=['Date']).set_index('Date')
        if 'Actual' not in df.columns:
            continue
        y = df['Actual'].values.astype(float)
        if np.sum(np.isfinite(y)) < 10:
            continue
        low, med, high = _bucket_masks_by_quantiles(y, qlow, qhigh)

        for m in models:
            col = f'Predicted_{m}'
            if col not in df.columns:
                continue
            yhat = df[col].values.astype(float)
            mask = np.isfinite(y) & np.isfinite(yhat)
            if np.sum(mask) < 10:
                continue
            # per-timestamp SMAPE%
            sm = per_timestamp_smape_percent(y[mask], yhat[mask])
            # align masks to masked positions
            low_m = low[mask]; med_m = med[mask]; high_m = high[mask]
            def _safe_mean(arr, msk):
                vals = arr[msk]
                return float(np.mean(vals)) if vals.size else np.nan
            sm_low = _safe_mean(sm, low_m)
            sm_med = _safe_mean(sm, med_m)
            sm_high = _safe_mean(sm, high_m)
            per_asset_bucket.append({
                'asset': asset,
                'model': m,
                'smape_low_pct': sm_low,
                'smape_med_pct': sm_med,
                'smape_high_pct': sm_high,
            })

    dfb = pd.DataFrame(per_asset_bucket)
    if dfb.empty:
        return pd.DataFrame(columns=[
            'model','smape_low_pct_mean','smape_low_pct_se',
            'smape_med_pct_mean','smape_med_pct_se',
            'smape_high_pct_mean','smape_high_pct_se'
        ])

    rows = []
    for m, sub in dfb.groupby('model'):
        low_vals = sub['smape_low_pct'].values.astype(float)
        med_vals = sub['smape_med_pct'].values.astype(float)
        high_vals = sub['smape_high_pct'].values.astype(float)
        rows.append({
            'model': m,
            'smape_low_pct_mean': float(np.nanmean(low_vals)),
            'smape_low_pct_se': float(_se(low_vals)),
            'smape_med_pct_mean': float(np.nanmean(med_vals)),
            'smape_med_pct_se': float(_se(med_vals)),
            'smape_high_pct_mean': float(np.nanmean(high_vals)),
            'smape_high_pct_se': float(_se(high_vals)),
        })
    out = pd.DataFrame(rows).sort_values('model')
    return out

def save_table_regimes(df: pd.DataFrame, outfile_csv: str):
    os.makedirs(os.path.dirname(outfile_csv), exist_ok=True)
    df.to_csv(outfile_csv, index=False)
