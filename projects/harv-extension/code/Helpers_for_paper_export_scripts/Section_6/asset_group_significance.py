"""
Asset-group significance (Milestone 6)

Reads per-asset predictions written by your runners and produces group-level
significance tables comparing explicit models vs a baseline using DM tests.

Inputs:
  --pred-dir           Directory with per-asset predictions CSVs (intraday or daily)
  --baseline           Baseline model code (default HAR)
  --models             Comma-separated comparison models (e.g., PM,CP)
  --groups-file        Optional CSV mapping tickers to groups with columns: ticker,group
  --tickers            Optional comma-separated tickers to restrict the analysis
  --out-long           Output CSV (long format)
  --out-pivot          Output CSV (pivot table: rows=group, cols=model, values=Fisher p)
  --min-points         Minimum aligned points per asset to include (default 200)

Notes:
- DM test uses |e| loss (MAE-style) with Newey–West variance (via utils.metrics_utils.dm_test)
- Effect sizes are baseline − model, so positive means the model improves over baseline
"""

import os
import argparse
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.metrics_utils import compute_metrics, dm_test, fisher_p

# ---------- Default grouping: ticker -> group ----------
TICKER_TO_GROUP = {
    # Big Tech / US Mega Growth
    "AAPL": "Big Tech", "MSFT": "Big Tech", "NVDA": "Big Tech",
    "GOOGL": "Big Tech", "META": "Big Tech", "AMZN": "Big Tech", "TSLA": "Big Tech",

    # US Broad Equity
    "SPY": "US Broad Equity", "QQQ": "US Broad Equity", "RSP": "US Broad Equity",

    # Leveraged Equity ETFs
    "SPXL": "Leveraged Equity", "TQQQ": "Leveraged Equity", "SOXL": "Leveraged Equity",

    # Semiconductors
    "SMH": "Semiconductors",

    # Sector ETFs
    "XLE": "Energy Sector", "XLF": "Financials Sector", "XLK": "Tech Sector",
    "XLU": "Utilities Sector", "XBI": "Biotech Sector",

    # International / EM Equity
    "EEM": "International/EM", "EFA": "International/EM", "INDA": "International/EM",
    "EWW": "International/EM", "EWZ": "International/EM",

    # China Equity / Internet
    "FXI": "China Equity", "KWEB": "China Equity", "BABA": "China Equity",

    # Commodities
    "GLD": "Commodities", "SLV": "Commodities", "USO": "Commodities", "DBB": "Commodities",

    # Currency / Crypto proxy
    "UUP": "Currency/Dollar", "FXY": "Currency/Yen", "BITO": "Crypto Proxy",

    # Fixed Income / Vol
    "TLT": "Fixed Income", "HYG": "Fixed Income", "VXX": "Volatility ETN",

    # Thematic / Other
    "ARKK": "Thematic/Innovation", "ARKW": "Thematic/Innovation", "MJ": "Thematic/Cannabis",

    # Single Stocks (Non-Big-Tech)
    "JNJ": "Single Stocks (Defensive)", "JPM": "Single Stocks (Financials)", "BTI": "Single Stocks (Staples)",

    # Palantir (growth/AI but not in Big Tech bucket)
    "PLTR": "US Growth (AI)"
}

def load_groups(groups_file: str | None, restrict: list[str] | None) -> dict:
    if groups_file and os.path.exists(groups_file):
        gdf = pd.read_csv(groups_file)
        if not {'ticker', 'group'}.issubset(set(gdf.columns.str.lower())):
            raise SystemExit("groups-file must have columns: ticker, group")
        # normalize
        m = {}
        for _, r in gdf.iterrows():
            t = str(r['ticker']).strip().upper()
            g = str(r['group']).strip()
            m[t] = g
        if restrict:
            return {t: m[t] for t in restrict if t in m}
        return m
    # default mapping
    if restrict:
        return {t: TICKER_TO_GROUP[t] for t in restrict if t in TICKER_TO_GROUP}
    return TICKER_TO_GROUP.copy()

def read_predictions(pred_dir: str, tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        path = os.path.join(pred_dir, f"{t}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['Date']).set_index('Date').sort_index()
            out[t] = df
    return out

def per_asset_stats(df: pd.DataFrame, baseline: str, model: str, min_points: int):
    y = df['Actual'].values.astype(float)
    base_col = f'Predicted_{baseline}'
    model_col = f'Predicted_{model}'
    if base_col not in df.columns or model_col not in df.columns:
        return None
    yb = df[base_col].values.astype(float)
    ym = df[model_col].values.astype(float)
    mask = np.isfinite(y) & np.isfinite(yb) & np.isfinite(ym)
    y, yb, ym = y[mask], yb[mask], ym[mask]
    if y.size < min_points:
        return None

    mets_base = compute_metrics(y, yb)
    mets_model = compute_metrics(y, ym)

    # improvements: baseline − model (positive is better)
    d_smape = mets_base['smape_pct'] - mets_model['smape_pct']
    d_mae   = mets_base['mae']       - mets_model['mae']
    d_rmse  = mets_base['rmse']      - mets_model['rmse']

    # DM on |e|
    loss_a = np.abs(y - yb)
    loss_b = np.abs(y - ym)
    _, p_dm = dm_test(loss_a, loss_b)

    # also record a "win" on SMAPE
    win_smape = 1.0 if d_smape > 0 else 0.0

    return {
        'delta_smape_pct': float(d_smape),
        'delta_mae': float(d_mae),
        'delta_rmse': float(d_rmse),
        'dm_p_mae': float(p_dm) if p_dm is not None else np.nan,
        'win_smape': float(win_smape),
        'n': int(y.size),
    }

def group_aggregate(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[
            'group','baseline','model','n_assets',
            'delta_smape_pct_mean','delta_smape_pct_se',
            'delta_mae_mean','delta_mae_se',
            'delta_rmse_mean','delta_rmse_se',
            'winner_frac_smape','fisher_p_dm','sig_mark'
        ])

    def _se(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size <= 1: return np.nan
        return float(np.nanstd(arr, ddof=1) / np.sqrt(np.sum(np.isfinite(arr))))

    out_rows = []
    for (g, b, m), sub in df.groupby(['group','baseline','model']):
        dsm = sub['delta_smape_pct'].values.astype(float)
        dma = sub['delta_mae'].values.astype(float)
        drm = sub['delta_rmse'].values.astype(float)
        wins = sub['win_smape'].values.astype(float)
        pvals = sub['dm_p_mae'].values.astype(float)

        fisher = fisher_p(pvals)
        # star only if mean effect is beneficial and p small
        sig_mark = ''
        if np.nanmean(dsm) > 0:
            if np.isfinite(fisher) and fisher < 1e-3: sig_mark = '★'
            elif np.isfinite(fisher) and fisher < 1e-2: sig_mark = '‡'
            elif np.isfinite(fisher) and fisher < 5e-2: sig_mark = '†'

        out_rows.append({
            'group': g,
            'baseline': b,
            'model': m,
            'n_assets': int(sub['asset'].nunique()),
            'delta_smape_pct_mean': float(np.nanmean(dsm)),
            'delta_smape_pct_se': float(_se(dsm)),
            'delta_mae_mean': float(np.nanmean(dma)),
            'delta_mae_se': float(_se(dma)),
            'delta_rmse_mean': float(np.nanmean(drm)),
            'delta_rmse_se': float(_se(drm)),
            'winner_frac_smape': float(np.nanmean(wins)),
            'fisher_p_dm': float(fisher) if fisher is not None else np.nan,
            'sig_mark': sig_mark,
        })
    return pd.DataFrame(out_rows).sort_values(['group','model'])

def main():
    ap = argparse.ArgumentParser(description="Asset-group significance vs baseline (DM tests + effect sizes)")
    ap.add_argument('--pred-dir', type=str, required=True, help='Directory with per-asset predictions CSVs')
    ap.add_argument('--baseline', type=str, default='HAR', help='Baseline model code (default HAR)')
    ap.add_argument('--models', type=str, required=True, help='Comma-separated comparison models (e.g., PM,CP)')
    ap.add_argument('--groups-file', type=str, default='', help='Optional CSV: ticker,group to override defaults')
    ap.add_argument('--tickers', type=str, default='', help='Optional comma-separated ticker list to restrict')
    ap.add_argument('--out-long', type=str, required=True, help='Output CSV (long format)')
    ap.add_argument('--out-pivot', type=str, required=True, help='Output CSV (pivot table: Fisher p)')
    ap.add_argument('--min-points', type=int, default=200, help='Minimum aligned samples per asset to include')
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    if not models:
        raise SystemExit("Provide at least one model via --models.")

    requested = [t.strip().upper() for t in args.tickers.split(',') if t.strip()] if args.tickers else None
    # prefer only tickers with prediction files actually present
    avail = []
    if os.path.isdir(args.pred_dir):
        avail = [os.path.splitext(f)[0] for f in os.listdir(args.pred_dir) if f.endswith('.csv')]
    tickers = requested if requested else avail
    if not tickers:
        raise SystemExit("No prediction CSVs found. Run the runners first.")

    t2g = load_groups(args.groups_file, restrict=tickers)
    if not t2g:
        raise SystemExit("No tickers matched the group mapping.")

    preds = read_predictions(args.pred_dir, tickers)
    rows = []
    for t in tickers:
        if t not in preds or t not in t2g:
            continue
        df = preds[t]
        for m in models:
            res = per_asset_stats(df, args.baseline, m, args.min_points)
            if res is None:
                continue
            rows.append({
                'asset': t,
                'group': t2g[t],
                'baseline': args.baseline,
                'model': m,
                **res
            })

    if not rows:
        raise SystemExit("No valid assets after filtering and model availability checks.")

    long_df = pd.DataFrame(rows).sort_values(['group','asset','model'])
    os.makedirs(os.path.dirname(args.out_long), exist_ok=True)
    long_df.to_csv(args.out_long, index=False)

    grp_df = group_aggregate(rows)
    # pivot table: Fisher p per group x model
    pivot = grp_df.pivot(index='group', columns='model', values='fisher_p_dm').sort_index()
    os.makedirs(os.path.dirname(args.out_pivot), exist_ok=True)
    pivot.to_csv(args.out_pivot)

    # Also drop a group-level long table next to pivot for convenience
    base = os.path.splitext(args.out_pivot)[0]
    grp_df.to_csv(base + "_long.csv", index=False)

    print(f"Wrote {args.out_long}")
    print(f"Wrote {args.out_pivot}")
    print(f"Wrote {base + '_long.csv'}")

if __name__ == '__main__':
    main()


