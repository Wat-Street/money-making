# code/paper_export/table_1_overall.py
# Generates LaTeX (booktabs) for Table 1A/1B with:
# - SMAPE, RMSE, MAE, DirAcc, MedAE, N
# - Bold best, underline second-best
# - DM marks on SMAPE vs HAR (â€  â€¡ â˜…) from sig_mark_smape in CSV
# - PM/CP core name gets a dagger if any variant row included has strictly lower SMAPE mean than the core
#
# Usage:
#   python code/paper_export/table_1_overall.py \
#     --csv outputs/intraday/tables/table_1a_overall.csv \
#     --out paper/latex/table_1a_overall.tex \
#     --context intraday \
#     --include-variants            # default on
#   # or exclude variants:
#   python code/paper_export/table_1_overall.py --csv ... --out ... --no-variants

import os
import argparse
import numpy as np
import pandas as pd

DISPLAY_ORDER = [
    'HAR','HAR_J','HAR_CJ','HAR_TCJ',
    'PM','PM_VW','PM_AD',
    'CP','CP_CJ',
    'EXH','HAM','RAND'
]

DISPLAY_NAME = {
    'HAR': 'HAR-RV',
    'HAR_J': 'HAR-RV-J',
    'HAR_CJ': 'HAR-RV-CJ',
    'HAR_TCJ': 'HAR-RV-TCJ',
    'PM': 'Prime Modulo (PM)',
    'PM_VW': 'Prime Modulo (PM-VW)',
    'PM_AD': 'Prime Modulo (PM-AD)',
    'CP': 'Contiguous Prime (CP)',
    'CP_CJ': 'Contiguous Prime (CP-CJ)',
    'EXH': 'Exhaustive Prefixes (EXH)',
    'HAM': 'Hamming Codes (HAM)',
    'RAND': 'Randomized (control)',
}

def _pm(s, se, digits=2):
    if not np.isfinite(s): return '--'
    if not np.isfinite(se): se = 0.0
    return f'{s:.{digits}f} $\\pm$ {se:.{digits}f}'

def _sci(x, se, sig=2):
    if not np.isfinite(x): return '--'
    if not np.isfinite(se): se = 0.0
    return f'{x:.{sig}e} $\\pm$ {se:.{max(1,sig-1)}e}'

def _bold2(series, higher_is_better=False):
    vals = np.array([np.nan] * len(series), dtype=float)
    for i, v in enumerate(series.index):
        try:
            # strip formatting to detect numeric ordering; fallback if needed
            raw = series.iloc[i]
            # extract number before space; robust for our "a Â± b" strings
            num = raw.split(' ')[0].replace('\\textbf{','').replace('\\underline{','').replace('}','')
            vals[i] = float(num) if 'e' in num or '.' in num or num.isdigit() else np.nan
        except Exception:
            vals[i] = np.nan
    order = np.argsort(-vals) if higher_is_better else np.argsort(vals)
    # mask invalid rows
    valid = np.isfinite(vals)
    if valid.sum() < 1:
        return series
    ranked = [i for i in order if valid[i]]
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    out = []
    for i, s in enumerate(series.tolist()):
        if i == best:
            out.append(r'\\textbf{' + s + '}')
        elif second is not None and i == second:
            out.append(r'\\underline{' + s + '}')
        else:
            out.append(s)
    return pd.Series(out, index=series.index)

def _dagger_core_names(df, include_variants):
    pm_suf = ''
    cp_suf = ''
    if include_variants:
        # PM core vs {PM_VW, PM_AD}
        if 'PM' in df['model'].values:
            core = float(df.loc[df['model']=='PM','smape_pct_mean'].iloc[0])
            cand = []
            if 'PM_VW' in df['model'].values:
                cand.append(float(df.loc[df['model']=='PM_VW','smape_pct_mean'].iloc[0]))
            if 'PM_AD' in df['model'].values:
                cand.append(float(df.loc[df['model']=='PM_AD','smape_pct_mean'].iloc[0]))
            if len(cand) and min([c for c in cand if np.isfinite(c)]) < core:
                pm_suf = r'$^\dagger$'
        # CP core vs {CP_CJ}
        if 'CP' in df['model'].values and 'CP_CJ' in df['model'].values:
            if float(df.loc[df['model']=='CP_CJ','smape_pct_mean'].iloc[0]) < float(df.loc[df['model']=='CP','smape_pct_mean'].iloc[0]):
                cp_suf = r'$^\dagger$'
    return pm_suf, cp_suf

def build_table(df, context, include_variants=True):
    # filter models
    keep = DISPLAY_ORDER if include_variants else [
        'HAR','HAR_J','HAR_CJ','HAR_TCJ','PM','CP','EXH','HAM','RAND'
    ]
    df = df[df['model'].isin(keep)].copy()
    df['model'] = pd.Categorical(df['model'], categories=DISPLAY_ORDER, ordered=True)
    df = df.sort_values('model')

    pm_suf, cp_suf = _dagger_core_names(df, include_variants)

    names = []
    for m in df['model'].tolist():
        n = DISPLAY_NAME.get(m, m)
        if m == 'PM' and pm_suf: n += pm_suf
        if m == 'CP' and cp_suf: n += cp_suf
        names.append(n)

    smape = df.apply(lambda r: _pm(r['smape_pct_mean'], r['smape_pct_se'], digits=2), axis=1)
    # add DM significance marks already computed vs HAR (improvement-only policy baked in CSV)
    sig = df['sig_mark_smape'].fillna('').astype(str)
    smape = smape + sig

    rmse  = df.apply(lambda r: _sci(r['rmse_mean'],  r['rmse_se'],  sig=2), axis=1)
    mae   = df.apply(lambda r: _sci(r['mae_mean'],   r['mae_se'],   sig=2), axis=1)
    dirac = df.apply(lambda r: _pm(r['diracc_pct_mean'], r['diracc_pct_se'], digits=2), axis=1)

    have_medae = set(['medae_mean','medae_se']).issubset(df.columns)
    medae = df.apply(lambda r: _sci(r['medae_mean'], r['medae_se'], sig=2), axis=1) if have_medae else None

    have_n = 'n_test_mean' in df.columns
    nvals = df['n_test_mean'].round(0).astype(int).astype(str) if have_n else pd.Series(['--']*len(df))

    # best / second best
    smape = _bold2(smape, higher_is_better=False)
    rmse  = _bold2(rmse,  higher_is_better=False)
    mae   = _bold2(mae,   higher_is_better=False)
    dirac = _bold2(dirac, higher_is_better=True)
    if have_medae: medae = _bold2(medae, higher_is_better=False)

    # assemble LaTeX
    caption = ('Table 1A (Intraday, 5-min): Overall forecasting performance. '
               'Lower is better except Directional Accuracy.')
    if context == 'daily':
        caption = 'Table 1B (Daily): Overall forecasting performance. Lower is better except Directional Accuracy.'
    label = 'tab:overall_intraday' if context == 'intraday' else 'tab:overall_daily'

    cols = [
        ('Model','names'),
        (r'SMAPE (\%)','smape'),
        ('RMSE','rmse'),
        ('MAE','mae'),
        (r'Dir. Acc. (\%)','dirac'),
    ]
    if have_medae:
        cols.append(('MedAE','medae'))
    cols.append(('N','nvals'))

    disp = pd.DataFrame({
        'names': names,
        'smape': smape,
        'rmse': rmse,
        'mae': mae,
        'dirac': dirac,
        'nvals': nvals
    })
    if have_medae:
        disp['medae'] = medae

    colspec = 'l' + 'c'*(len(cols)-1)
    lines = []
    lines += [r'\begin{table}[t]', r'\centering',
              r'\caption{' + caption + r'}',
              r'\label{' + label + r'}',
              r'\begin{tabular}{' + colspec + r'}',
              r'\toprule']
    lines += [' & '.join([c[0] for c in cols]) + r' \\', r'\midrule']
    for _, row in disp.iterrows():
        parts = [row[c[1]] for c in cols]
        lines += [' & '.join(parts) + r' \\']
    lines += [r'\bottomrule', r'\end{tabular}']
    notes = (r'\vspace{2pt}\par\footnotesize '
             r'Notes: Mean across assets; $\,\pm\,$ is standard error across assets. '
             r'Bold = best, underline = second-best per column. '
             r'Superscripts on SMAPE denote Diebold--Mariano significance vs HAR-RV '
             r'($\dagger$: $p<0.05$, $\ddagger$: $p<0.01$, $\star$: $p<0.001$).')
    notes += (r' If present, $^\dagger$ after PM/CP indicates a variant row in this table beats the core; '
              r'see also ablations.')
    lines += [notes, r'\end{table}']
    return '\n'.join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--context', choices=['intraday','daily'], default='intraday')
    ap.add_argument('--include-variants', dest='include_variants', action='store_true', default=True)
    ap.add_argument('--no-variants', dest='include_variants', action='store_false')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    tex = build_table(df, context=args.context, include_variants=args.include_variants)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(tex)
    print(f'[ok] wrote {args.out}')

if __name__ == '__main__':
    main()


