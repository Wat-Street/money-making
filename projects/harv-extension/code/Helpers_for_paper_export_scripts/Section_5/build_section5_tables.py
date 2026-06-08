#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Section 5 tables from intraday predictions.

Inputs (discovered):
  code/outputs/intraday/predictions/{TICKER}.csv
    columns: Date, Actual, Predicted_<MODEL> (for each model)

Outputs (under --tables-dir):
  Section_5_table_5a_pm_ablations.csv
  Section_5_table_5b_cp_ablations.csv
  Section_5_table_4_random_controls.csv
  Section_5_table_3_param_eff_joined.csv

Notes
- ΔSMAPE := SMAPE_base − SMAPE_variant (pp). Positive => variant improves.
- ΔMAE   := MAE_base − MAE_variant. Positive => variant improves.
- ΔRMSE  := RMSE_base − RMSE_variant. Positive => variant improves.
- ΔDA    := DirAcc_variant − DirAcc_base (pp). Positive => variant improves.
- DM tests:
    * SMAPE: use pointwise SMAPE differential
    * MAE:   use pointwise |err| differential
    * RMSE:  use pointwise squared-error differential (loss); report scalar ΔRMSE
    * DirAcc: two-proportion z-test per asset, then Fisher-combine across assets.
"""
import os
import argparse
import numpy as np
import pandas as pd
from math import erf, sqrt, log

try:
    from scipy.stats import norm, chi2
    _has_scipy = True
except Exception:
    _has_scipy = False

EPS = 1e-12
TIE_EPS = 1e-12  # treat |delta| < TIE_EPS as tie when computing win fraction

DISPLAY = {
    'HAR':'HAR-RV','HAR_J':'HAR-RV-J','HAR_CJ':'HAR-RV-CJ','HAR_TCJ':'HAR-RV-TCJ',
    'PM':'Prime Modulo (PM)','PM_VW':'PM-VW','PM_AD':'PM-AD',
    'CP':'Contiguous Prime (CP)','CP_CJ':'CP-CJ',
    'EXH':'Exhaustive Prefixes (EXH)','HAM':'Hamming Codes (HAM)',
    'RAND':'Random (control)','CRS':'Contiguous Random (control)'
}

# ---------- helpers ----------

def smape_series(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return 200.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + EPS)

def mae_series(y_true, y_pred):
    return np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))

def sqerr_series(y_true, y_pred):
    e = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return e * e

def dm_test(delta):
    """Diebold–Mariano (NW variance, 2-sided)."""
    d = np.asarray(delta, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 20:
        return np.nan
    dbar = d.mean()
    L = int(np.floor(1.5 * n ** (1/3)))
    dc = d - dbar
    gamma0 = np.dot(dc, dc) / n
    var_hat = gamma0
    for k in range(1, min(L, n-1)+1):
        w = 1.0 - k / (L+1.0)
        cov = np.dot(dc[:-k], dc[k:]) / n
        var_hat += 2.0 * w * cov
    var_mean = var_hat / n
    if var_mean <= 0 or not np.isfinite(var_mean):
        return np.nan
    t = dbar / np.sqrt(var_mean)
    if _has_scipy:
        from scipy.stats import norm
        return float(2.0 * (1.0 - norm.cdf(abs(t))))
    # normal CDF via erf
    return float(2.0 * (1.0 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))

def fisher_combine(pvals):
    pvals = [p for p in pvals if np.isfinite(p) and p>0]
    if not pvals:
        return np.nan
    stat = -2.0 * sum(log(p) for p in pvals)
    df = 2*len(pvals)
    if _has_scipy:
        from scipy.stats import chi2
        return float(1.0 - chi2.cdf(stat, df))
    # rough fallback
    z = (stat - df) / sqrt(2*df)
    if _has_scipy:
        from scipy.stats import norm
        return float(1.0 - norm.cdf(z))
    return np.nan

def two_prop_z_test(c1, n1, c2, n2):
    """2-proportion z-test (two-sided). Returns p-value."""
    if n1 <= 0 or n2 <= 0:
        return np.nan
    p1 = c1 / max(1, n1); p2 = c2 / max(1, n2)
    p_pool = (c1 + c2) / max(1, (n1 + n2))
    denom = np.sqrt(p_pool * (1 - p_pool) * (1.0/n1 + 1.0/n2) + 1e-18)
    if denom == 0:
        return np.nan
    z = (p1 - p2) / denom
    if _has_scipy:
        from scipy.stats import norm
        return float(2.0 * (1.0 - norm.cdf(abs(z))))
    return float(2.0 * (1.0 - 0.5 * (1 + erf(abs(z) / sqrt(2)))))

def mean_se(xs):
    xs = np.asarray(xs, float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0: 
        return np.nan, np.nan
    m = xs.mean()
    se = xs.std(ddof=1)/np.sqrt(xs.size) if xs.size>1 else np.nan
    return float(m), (float(se) if np.isfinite(se) else np.nan)

def clean_small(x, thr=1e-12):
    try:
        xv = float(x)
        return 0.0 if abs(xv) < thr else xv
    except Exception:
        return x

# ---------- IO ----------

def load_predictions(pred_dir, ticker):
    path = os.path.join(pred_dir, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    dt_col = 'Date' if 'Date' in df.columns else ('date' if 'date' in df.columns else None)
    if dt_col:
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.set_index(dt_col).sort_index()
    return df

# ---------- per-asset metrics ----------

def diracc_pct(y_true, y_pred):
    """Directional accuracy on first differences (percentage in [0,100])."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    dy_t = np.diff(y_true)
    dy_p = np.diff(y_pred)
    ok = np.isfinite(dy_t) & np.isfinite(dy_p)
    if ok.sum() == 0:
        return np.nan, 0, 0
    s_t = np.sign(dy_t[ok])
    s_p = np.sign(dy_p[ok])
    hits = (s_t == s_p).sum()
    total = ok.sum()
    return 100.0 * hits / total, int(hits), int(total)

def per_asset_metrics(df, models):
    y = df['Actual'].values.astype(float)
    out = {}
    for m in models:
        col = f"Predicted_{m}"
        if col not in df.columns: 
            continue
        yhat = df[col].values.astype(float)
        s_smape = smape_series(y, yhat)     # %
        s_mae   = mae_series(y, yhat)       # abs error
        s_sqerr = sqerr_series(y, yhat)     # squared error
        rmse    = float(np.sqrt(np.nanmean(s_sqerr)))
        mae     = float(np.nanmean(s_mae))
        smape   = float(np.nanmean(s_smape))
        da_pct, da_hits, da_total = diracc_pct(y, yhat)
        out[m] = {
            'smape_mean': smape,
            'mae_mean': mae,
            'rmse_mean': rmse,
            'smape_series': s_smape,
            'mae_series': s_mae,
            'sqerr_series': s_sqerr,
            'diracc_pct': da_pct,
            'diracc_hits': da_hits,
            'diracc_total': da_total,
        }
    return out

# ---------- builders ----------

def _aggregate_summary(df: pd.DataFrame) -> dict:
    out = {}

    # numeric means & SEs (separate assignment; force to float; clean small)
    for col in ['delta_smape_pct','delta_mae','delta_rmse','delta_diracc_pct']:
        if col in df.columns:
            m, s = mean_se(df[col].values)
            out[f'{col}_mean'] = clean_small(m)
            out[f'{col}_se']   = (float(s) if np.isfinite(s) else np.nan)

    # win fraction on SMAPE, treating near-zero as tie
    if 'delta_smape_pct' in df.columns:
        d = np.asarray(df['delta_smape_pct'].values, float)
        wins = (d > TIE_EPS).sum()
        trials = np.isfinite(d).sum()
        out['winner_frac_smape'] = float(wins / trials) if trials > 0 else np.nan

    # fisher combine (per metric)
    def fisher(colname):
        return fisher_combine(df.get(colname, pd.Series(dtype=float)).tolist())

    out['fisher_p_dm_smape'] = fisher('dm_p_smape')
    out['fisher_p_dm_mae']   = fisher('dm_p_mae')
    out['fisher_p_dm_rmse']  = fisher('dm_p_rmse')
    out['fisher_p_diracc']   = fisher('p_diracc')

    def sig(p):
        if not np.isfinite(p):
            return ''
        return '?' if p < 1e-3 else ('Ø' if p < 1e-2 else ('+' if p < 5e-2 else ''))
    out['sig_mark_smape'] = sig(out['fisher_p_dm_smape'])
    out['sig_mark_mae']   = sig(out['fisher_p_dm_mae'])
    out['sig_mark_rmse']  = sig(out['fisher_p_dm_rmse'])
    out['sig_mark_diracc']= sig(out['fisher_p_diracc'])

    # aliases for scripts expecting generic names
    out['fisher_p_dm'] = out['fisher_p_dm_smape']
    out['sig_mark'] = out['sig_mark_smape']

    if 'asset' in df.columns:
        out['n_assets'] = int(pd.Series(df['asset']).nunique())
    else:
        out['n_assets'] = int(len(df))
    return out


def aggregate_and_write(rows, out_csv):
    if not rows:
        return
    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if 'model' in df.columns:
        out_rows = []
        for model, sub in df.groupby('model'):
            summary = _aggregate_summary(sub)
            summary['model'] = model
            out_rows.append(summary)
        pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    else:
        summary = _aggregate_summary(df)
        pd.DataFrame([summary]).to_csv(out_csv, index=False)

def build_ablations(pred_dir, tables_dir, models, rand_baseline):
    os.makedirs(tables_dir, exist_ok=True)
    assets = sorted([f[:-4] for f in os.listdir(pred_dir) if f.endswith('.csv')])

    pm_rows, cp_rows, rand_rows = [], [], []

    for asset in assets:
        df = load_predictions(pred_dir, asset)
        if df is None or 'Actual' not in df.columns:
            continue
        mets = per_asset_metrics(df, models)

        # PM family. Write every available PM modifier instead of requiring
        # the whole modifier set; some runs may omit PM_AD but still include
        # PM_VW, and the section should still render the available ablation.
        if 'PM' in mets:
            base = mets['PM']
            for variant in ('PM_VW','PM_AD'):
                if variant not in mets:
                    continue
                var = mets[variant]
                d_smape = base['smape_series'] - var['smape_series']
                d_mae   = base['mae_series']   - var['mae_series']
                d_sqerr = base['sqerr_series'] - var['sqerr_series']  # RMSE via SE loss
                delta_rmse = base['rmse_mean'] - var['rmse_mean']
                delta_mae  = base['mae_mean']  - var['mae_mean']
                # directional accuracy (variant − base, pp)
                da_b = base['diracc_pct']; da_v = var['diracc_pct']
                dh_b, nt_b = base['diracc_hits'], base['diracc_total']
                dh_v, nt_v = var['diracc_hits'],  var['diracc_total']
                delta_da = (da_v - da_b) if (np.isfinite(da_b) and np.isfinite(da_v)) else np.nan
                p_da = two_prop_z_test(dh_v, nt_v, dh_b, nt_b)

                pm_rows.append({
                    'asset': asset, 'model': variant,
                    'delta_smape_pct': float(np.nanmean(d_smape)),
                    'delta_mae': float(delta_mae),
                    'delta_rmse': float(delta_rmse),
                    'delta_diracc_pct': float(delta_da) if np.isfinite(delta_da) else np.nan,
                    'dm_p_smape': dm_test(d_smape),
                    'dm_p_mae': dm_test(d_mae),
                    'dm_p_rmse': dm_test(d_sqerr),
                    'p_diracc': p_da,
                })

        # CP family
        if all(k in mets for k in ('CP','CP_CJ')):
            base = mets['CP']; var = mets['CP_CJ']
            d_smape = base['smape_series'] - var['smape_series']
            d_mae   = base['mae_series']   - var['mae_series']
            d_sqerr = base['sqerr_series'] - var['sqerr_series']
            delta_rmse = base['rmse_mean'] - var['rmse_mean']
            delta_mae  = base['mae_mean']  - var['mae_mean']
            da_b = base['diracc_pct']; da_v = var['diracc_pct']
            dh_b, nt_b = base['diracc_hits'], base['diracc_total']
            dh_v, nt_v = var['diracc_hits'],  var['diracc_total']
            delta_da = (da_v - da_b) if (np.isfinite(da_b) and np.isfinite(da_v)) else np.nan
            p_da = two_prop_z_test(dh_v, nt_v, dh_b, nt_b)

            cp_rows.append({
                'asset': asset, 'model': 'CP_CJ',
                'delta_smape_pct': float(np.nanmean(d_smape)),
                'delta_mae': float(delta_mae),
                'delta_rmse': float(delta_rmse),
                'delta_diracc_pct': float(delta_da) if np.isfinite(delta_da) else np.nan,
                'dm_p_smape': dm_test(d_smape),
                'dm_p_mae': dm_test(d_mae),
                'dm_p_rmse': dm_test(d_sqerr),
                'p_diracc': p_da,
            })

        # RAND / CRS vs baseline (default PM)
        base_name = rand_baseline
        for ctrl in ('RAND', 'CRS'):
            if not all(k in mets for k in (ctrl, base_name)):
                continue
            base = mets[base_name]; var = mets[ctrl]
            d_smape = base['smape_series'] - var['smape_series']
            d_mae   = base['mae_series']   - var['mae_series']
            d_sqerr = base['sqerr_series'] - var['sqerr_series']
            delta_rmse = base['rmse_mean'] - var['rmse_mean']
            delta_mae  = base['mae_mean']  - var['mae_mean']
            da_b = base['diracc_pct']; da_v = var['diracc_pct']
            dh_b, nt_b = base['diracc_hits'], base['diracc_total']
            dh_v, nt_v = var['diracc_hits'],  var['diracc_total']
            delta_da = (da_v - da_b) if (np.isfinite(da_b) and np.isfinite(da_v)) else np.nan
            p_da = two_prop_z_test(dh_v, nt_v, dh_b, nt_b)

            rand_rows.append({
                'asset': asset, 'model': ctrl,
                'delta_smape_pct': float(np.nanmean(d_smape)),
                'delta_mae': float(delta_mae),
                'delta_rmse': float(delta_rmse),
                'delta_diracc_pct': float(delta_da) if np.isfinite(delta_da) else np.nan,
                'dm_p_smape': dm_test(d_smape),
                'dm_p_mae': dm_test(d_mae),
                'dm_p_rmse': dm_test(d_sqerr),
                'p_diracc': p_da,
            })

    # write PM ablations (variants vs PM)
    if pm_rows:
        # aggregate per variant across assets
        aggregate_and_write(pm_rows, os.path.join(tables_dir, 'Section_5_table_5a_pm_ablations.csv'))

    # write CP ablations (CP_CJ vs CP)
    if cp_rows:
        aggregate_and_write(cp_rows, os.path.join(tables_dir, 'Section_5_table_5b_cp_ablations.csv'))

    # write RAND/CRS controls (vs baseline)
    if rand_rows:
        aggregate_and_write(rand_rows, os.path.join(tables_dir, 'Section_5_table_4_random_controls.csv'))

def build_param_eff(pred_dir, tables_dir, models):
    os.makedirs(tables_dir, exist_ok=True)
    assets = sorted([f[:-4] for f in os.listdir(pred_dir) if f.endswith('.csv')])

    rows = []
    for asset in assets:
        df = load_predictions(pred_dir, asset)
        if df is None or 'Actual' not in df.columns:
            continue
        mets = per_asset_metrics(df, models)
        for m, v in mets.items():
            rows.append({
                'asset': asset, 'model': m,
                'smape_pct': v['smape_mean'],
                'mae': v['mae_mean'],
                'rmse': v['rmse_mean'],
                'diracc_pct': v['diracc_pct'],
            })
    if not rows:
        return
    per_asset = pd.DataFrame(rows)

    def se_col(x): 
        x = pd.Series(x, dtype=float)
        if x.count() <= 1: 
            return np.nan
        return float(x.std(ddof=1) / np.sqrt(x.count()))

    agg = per_asset.groupby('model', as_index=False).agg(
        smape_pct_mean=('smape_pct','mean'),
        smape_pct_se=('smape_pct', se_col),
        mae_mean=('mae','mean'),
        mae_se=('mae', se_col),
        rmse_mean=('rmse','mean'),
        rmse_se=('rmse', se_col),
        diracc_pct_mean=('diracc_pct','mean'),
        diracc_pct_se=('diracc_pct', se_col),
    )

    # attach feature counts
    param_file = os.path.join(tables_dir, 'Section_5_table_3_param.csv')
    if os.path.exists(param_file):
        fc = pd.read_csv(param_file)
        feat = fc[['model','feature_count']].drop_duplicates()
    else:
        feat = pd.DataFrame({
            'model': ['HAR','HAR_J','HAR_CJ','HAR_TCJ','PM','PM_VW','PM_AD','CP','CP_CJ','EXH','HAM','RAND','CRS'],
            'feature_count': [3,9,9,9,6,6,6,6,6,15,12,6,6]
        })
    out = agg.merge(feat, on='model', how='left')
    out.to_csv(os.path.join(tables_dir, 'Section_5_table_3_param_eff_joined.csv'), index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', required=True)
    ap.add_argument('--tables-dir', required=True)
    ap.add_argument('--models', required=True, help='Comma list of models present in predictions')
    ap.add_argument('--rand-baseline', default='PM', help='Baseline for RAND control deltas (default PM)')
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    build_ablations(args.pred_dir, args.tables_dir, models, args.rand_baseline)
    build_param_eff(args.pred_dir, args.tables_dir, models)
    print("[ok] Section 5 tables refreshed.")

if __name__ == '__main__':
    main()
