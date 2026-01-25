import sys, os, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from utils.data_utils import fetch_intraday_data, fit_and_predict_extended
from utils.harvey_utils import (
    add_harv_terms, add_harv_j_terms, add_harv_cj_terms, add_harv_tcj_terms
)
from utils.models_utils import (
    add_exhaustive_terms, add_hamming_terms,
    add_prime_modulo_terms,
    add_volume_weighted_prime_modulo_terms,
    add_volume_weighted_adaptive_prime_modulo_terms,
    contig_prime_modulo, contig_prime_modulo_with_jumps,
    random_sets, contiguous_random_sets
)
from utils.reporting_utils import aggregate_overall_from_predictions, save_table_overall

MODEL_FUNCS = {
    'HAR': add_harv_terms,
    'HAR_J': add_harv_j_terms,
    'HAR_CJ': add_harv_cj_terms,
    'HAR_TCJ': add_harv_tcj_terms,
    'PM': add_prime_modulo_terms,
    'PM_VW': add_volume_weighted_prime_modulo_terms,
    'PM_AD': add_volume_weighted_adaptive_prime_modulo_terms,
    'CP': contig_prime_modulo,
    'CP_CJ': contig_prime_modulo_with_jumps,
    'EXH': add_exhaustive_terms,
    'HAM': add_hamming_terms,
    'RAND': random_sets,
    'CRS': contiguous_random_sets,
}

def _default_models(include_variants: bool = True):
    cores = ['HAR','HAR_J','HAR_CJ','HAR_TCJ','PM','CP','EXH','HAM','RAND','CRS']
    variants = ['PM_VW','PM_AD','CP_CJ']
    return cores + (variants if include_variants else [])

def discover_local_tickers(base_dir: str) -> list:
    if not os.path.isdir(base_dir):
        return []
    return sorted([f.replace('_5m.csv','') for f in os.listdir(base_dir) if f.endswith('_5m.csv')])

def intraday_to_daily(df5m: pd.DataFrame) -> pd.DataFrame:
    if 'Squared_Return' in df5m.columns:
        sr = df5m['Squared_Return'].astype(float)
    else:
        close = df5m['Close'].astype(float)
        ret = close.pct_change().fillna(0.0)
        sr = ret**2
    vol = df5m['Volume'] if 'Volume' in df5m.columns else pd.Series(0.0, index=df5m.index)

    g = df5m.groupby(df5m.index.date)
    sr_day = g.apply(lambda x: (sr.loc[x.index]).sum())
    vol_sum = g.apply(lambda x: (vol.loc[x.index]).sum())

    daily = pd.DataFrame({'SR': sr_day.values, 'Volume': vol_sum.values}, index=pd.to_datetime(sr_day.index))
    daily = daily.sort_index()
    # Realized volatility is sqrt of summed variance contributions
    daily['RV_d'] = np.sqrt(daily['SR'])
    daily['RV_w'] = np.sqrt(daily['SR'].rolling(window=5, min_periods=5).sum())
    daily['RV_m'] = np.sqrt(daily['SR'].rolling(window=22, min_periods=22).sum())
    return daily.dropna()

def run_for_ticker_daily(ticker: str, models: list, n: int, warmup: int, local_dir: str, outdir: str):
    os.makedirs(os.path.join(outdir, 'predictions'), exist_ok=True)

    df5m = fetch_intraday_data(ticker, use_local=True, local_dir=local_dir)
    if df5m is None or df5m.empty:
        print(f"[DAILY][skip] No data for {ticker}")
        return
    daily = intraday_to_daily(df5m)
    if daily is None or daily.empty:
        print(f"[DAILY][skip] No daily data after aggregation for {ticker}")
        return

    frames = []
    for m in models:
        if m not in MODEL_FUNCS:
            print(f"[DAILY][warn] Unknown model code: {m}")
            continue
        func = MODEL_FUNCS[m]
        extended = func(daily.copy(), n)
        # Include all engineered feature families (RV_*, PM_*, CP_*)
        features = [c for c in extended.columns if c.startswith(('RV', 'PM', 'CP'))]
        preds = fit_and_predict_extended(extended, features, n, warmup, model_name=m)
        if preds is None or preds.empty:
            print(f"[DAILY][warn] No predictions for {ticker} with {m}")
            continue
        if 'Predicted' in preds.columns and f'Predicted_{m}' not in preds.columns:
            preds[f'Predicted_{m}'] = preds['Predicted']
        cols = ['Actual', f'Predicted_{m}', f'Err_{m}', f'AbsErr_{m}', f'SMAPE_{m}_pct']
        frames.append(preds[[c for c in cols if c in preds.columns]])

    if not frames:
        print(f"[DAILY][skip] No valid model outputs for {ticker}")
        return

    merged = frames[0].copy()
    for f in frames[1:]:
        f2 = f.drop(columns=[c for c in ['Actual'] if c in f.columns])
        merged = merged.join(f2, how='inner')
    merged = merged.reset_index().rename(columns={'index': 'Date'})

    out_csv = os.path.join(outdir, 'predictions', f'{ticker}.csv')
    merged.to_csv(out_csv, index=False)
    print(f"[DAILY] wrote {out_csv}")

def main():
    p = argparse.ArgumentParser(description="Daily benchmark from local 5m CSVs (derive daily RV; run 12 models).")
    p.add_argument('--tickers', type=str, default='', help='Comma-separated tickers. If empty, auto-discover in --local-dir.')
    p.add_argument('--local-dir', type=str, default='Datasets/clean', help='Folder with *_5m.csv (relative to code/).')
    p.add_argument('--outdir', type=str, default='code/outputs/daily')
    p.add_argument('--models', type=str, default='', help='Comma-separated model codes. If empty, use defaults.')
    p.add_argument('--include-variants', dest='include_variants', action='store_true', default=True,
                   help='Include PM_VW, PM_AD, CP_CJ when --models not specified (default: on).')
    p.add_argument('--no-variants', dest='include_variants', action='store_false')
    p.add_argument('--n', type=int, default=22, help='Lookback horizon for RV_m and model-specific windows.')
    p.add_argument('--warmup', type=int, default=60, help='Minimum observations before starting OLS rolling forecasts.')
    p.add_argument('--require-explicit', action='store_true', help='Error if --tickers not provided (no auto-discovery).')
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()] if args.models.strip() else _default_models(include_variants=args.include_variants)

    tickers = [t.strip() for t in args.tickers.split(',') if t.strip()]
    if not tickers:
        if args.require_explicit:
            raise SystemExit("No --tickers provided and --require-explicit set. Pass a comma-separated list via --tickers.")
        tickers = discover_local_tickers(args.local_dir)

    os.makedirs(os.path.join(args.outdir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'tables'), exist_ok=True)

    for t in tickers:
        print(f'[DAILY] {t} :: models={",".join(models)}')
        run_for_ticker_daily(t, models, args.n, args.warmup, args.local_dir, args.outdir)

    overall, _ = aggregate_overall_from_predictions(os.path.join(args.outdir, 'predictions'), models)
    out_csv = os.path.join(args.outdir, 'tables', 'Section_1_table_1b_overall.csv')
    save_table_overall(overall, out_csv)
    print(f'[DAILY] wrote {out_csv}')

if __name__ == "__main__":
    main()


