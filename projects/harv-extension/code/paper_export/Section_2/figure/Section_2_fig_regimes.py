#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure: SMAPE by RV Regime (Intraday)
- Default: relative improvement vs HAR (%), so differences are readable.
- Option: --absolute to plot absolute SMAPE (%) instead.
- New: --scale MIN,MAX to clamp y-axis (e.g., --scale 0,50 for relative; --scale 100,220 for absolute).
- Exports vector graphics (PDF and SVG).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    'PM': 'PM', 'PM_VW': 'PM-VW', 'PM_AD': 'PM-AD',
    'CP': 'CP', 'CP_CJ': 'CP-CJ',
    'EXH': 'EXH', 'HAM': 'HAM', 'RAND': 'Random'
}

def _prep_df(df, include_variants: bool):
    keep = DISPLAY_ORDER if include_variants else [
        'HAR','HAR_J','HAR_CJ','HAR_TCJ','PM','CP','EXH','HAM','RAND'
    ]
    df = df[df['model'].isin(keep)].copy()
    df['model'] = pd.Categorical(df['model'], categories=DISPLAY_ORDER, ordered=True)
    df = df.sort_values('model').reset_index(drop=True)
    return df

def _delta_method_ratio(mean_m, se_m, mean_h, se_h):
    if not np.isfinite(mean_h) or mean_h == 0:
        return np.nan
    var_m = se_m**2
    var_h = se_h**2
    term = (var_m / (mean_h**2)) + ((mean_m**2) * var_h / (mean_h**4))
    return 100.0 * np.sqrt(term)

def make_relative(df):
    har = df[df['model'] == 'HAR']
    if har.empty:
        raise ValueError("HAR row not found in regimes CSV; relative plot needs HAR as baseline.")
    H_low,  H_low_se  = float(har['smape_low_pct_mean']),  float(har['smape_low_pct_se'])
    H_med,  H_med_se  = float(har['smape_med_pct_mean']),  float(har['smape_med_pct_se'])
    H_high, H_high_se = float(har['smape_high_pct_mean']), float(har['smape_high_pct_se'])

    rel = df.copy()
    def _rel(mean_m, se_m, mean_h, se_h):
        delta = 100.0 * (mean_h - mean_m) / mean_h
        se = _delta_method_ratio(mean_m, se_m, mean_h, se_h)
        return delta, se

    L = np.vectorize(lambda m, s: _rel(m, s, H_low,  H_low_se))(
        rel['smape_low_pct_mean'].values,  rel['smape_low_pct_se'].values)
    M = np.vectorize(lambda m, s: _rel(m, s, H_med,  H_med_se))(
        rel['smape_med_pct_mean'].values,  rel['smape_med_pct_se'].values)
    H = np.vectorize(lambda m, s: _rel(m, s, H_high, H_high_se))(
        rel['smape_high_pct_mean'].values, rel['smape_high_pct_se'].values)

    rel['d_low_mean']  = [x[0] for x in zip(*L)]
    rel['d_low_se']    = [x[1] for x in zip(*L)]
    rel['d_med_mean']  = [x[0] for x in zip(*M)]
    rel['d_med_se']    = [x[1] for x in zip(*M)]
    rel['d_high_mean'] = [x[0] for x in zip(*H)]
    rel['d_high_se']   = [x[1] for x in zip(*H)]
    return rel

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
    ax.tick_params(axis='x', labelrotation=30, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)

def _apply_scale(ax, scale):
    if scale is None:
        return
    try:
        lo, hi = [float(s) for s in scale.split(',')]
        ax.set_ylim(lo, hi)
    except Exception:
        pass  # ignore malformed scale strings

def plot_absolute(df, out_pdf, out_svg, scale):
    labels = [DISPLAY_NAME.get(m, m) for m in df['model']]
    x = np.arange(len(df))
    width = 0.25

    plt.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(11, 4.6))
    ax = plt.gca()

    ax.bar(x - width, df['smape_low_pct_mean'],  width,
           yerr=df['smape_low_pct_se'].fillna(0.0),  capsize=3, label='Low')
    ax.bar(x,          df['smape_med_pct_mean'],  width,
           yerr=df['smape_med_pct_se'].fillna(0.0), capsize=3, label='Med')
    ax.bar(x + width,  df['smape_high_pct_mean'], width,
           yerr=df['smape_high_pct_se'].fillna(0.0),capsize=3, label='High')

    ax.set_xticks(x); ax.set_xticklabels(labels, ha='right')
    ax.set_ylabel('SMAPE (%)')
    ax.set_title('SMAPE% by RV Regime (Intraday)')
    style_axes(ax)
    _apply_scale(ax, scale)
    ax.legend(ncol=3, frameon=False, fontsize=9)

    # regime note
    fig.text(0.99, 0.02, 'RV regimes = terciles by realized volatility (33rd/66th percentiles).',
             ha='right', va='bottom', fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_svg, bbox_inches='tight')
    plt.close(fig)

def plot_relative(df, out_pdf, out_svg, scale):
    rel = make_relative(df)
    labels = [DISPLAY_NAME.get(m, m) for m in rel['model']]
    x = np.arange(len(rel))
    width = 0.25

    plt.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(11, 4.6))
    ax = plt.gca()

    ax.axhline(0.0, linewidth=1.0)

    ax.bar(x - width, rel['d_low_mean'],  width,
           yerr=rel['d_low_se'].fillna(0.0),  capsize=3, label='Low')
    ax.bar(x,          rel['d_med_mean'],  width,
           yerr=rel['d_med_se'].fillna(0.0), capsize=3, label='Med')
    ax.bar(x + width,  rel['d_high_mean'], width,
           yerr=rel['d_high_se'].fillna(0.0),capsize=3, label='High')

    ax.set_xticks(x); ax.set_xticklabels(labels, ha='right')
    ax.set_ylabel('Î”SMAPE vs HAR (%)  (positive = better)')
    ax.set_title('Relative Improvement by RV Regime (Intraday)')
    style_axes(ax)
    _apply_scale(ax, scale)
    ax.legend(ncol=3, frameon=False, fontsize=9)

    fig.text(0.99, 0.02, 'RV regimes = terciles by realized volatility (33rd/66th percentiles).',
             ha='right', va='bottom', fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_svg, bbox_inches='tight')
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--outdir', required=True)
    # default is relative; use --absolute to switch
    ap.add_argument('--absolute', dest='relative', action='store_false',
                    help='Plot absolute SMAPE instead of relative improvement.')
    ap.add_argument('--relative', dest='relative', action='store_true', default=True)
    ap.add_argument('--include-variants', dest='include_variants', action='store_true', default=True)
    ap.add_argument('--no-variants', dest='include_variants', action='store_false')
    ap.add_argument('--scale', type=str, default=None,
                    help='Clamp y-axis to MIN,MAX (e.g., --scale 0,50 for relative; --scale 100,220 for absolute).')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = _prep_df(df, include_variants=args.include_variants)

    if args.relative:
        out_pdf = os.path.join(args.outdir, 'Section_2_fig_1_regimes_relative.pdf')
        out_svg = os.path.join(args.outdir, 'Section_2_fig_1_regimes_relative.svg')
        plot_relative(df, out_pdf, out_svg, args.scale)
    else:
        out_pdf = os.path.join(args.outdir, 'Section_2_fig_1_regimes_absolute.pdf')
        out_svg = os.path.join(args.outdir, 'Section_2_fig_1_regimes_absolute.svg')
        plot_absolute(df, out_pdf, out_svg, args.scale)

if __name__ == '__main__':
    main()



