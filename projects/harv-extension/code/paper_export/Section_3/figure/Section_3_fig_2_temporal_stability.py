#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure: Temporal Stability (Rolling SMAPE)
- Plots rolling SMAPE (%) over time for 2â€“3 selected models
- Professional styling (monochrome by default), clear dashes, thinner grid
- Optional smoothing, resampling (e.g., weekly mean), and linear trend
- Optional y-axis clamp via --scale MIN,MAX
- Exports PDF + SVG

Examples:
  python code/paper_export/Section_3_fig_temporal_stability.py \
    --pred-dir run_results/current_intraday/predictions \
    --asset SPY \
    --models HAR,PM,CP \
    --window 78 \
    --smooth ema --smooth-span 39 \
    --resample W \
    --trend \
    --scale 90,170 \
    --outdir run_results/current_intraday/figures

  # color style (instead of monochrome)
  --style color
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

EPS = 1e-12

DISPLAY_NAME = {
    'HAR': 'HAR-RV', 'HAR_J': 'HAR-RV-J', 'HAR_CJ': 'HAR-RV-CJ', 'HAR_TCJ': 'HAR-RV-TCJ',
    'PM': 'Prime Modulo (PM)', 'PM_VW': 'PM-VW', 'PM_AD': 'PM-AD',
    'CP': 'Contiguous Prime (CP)', 'CP_CJ': 'CP-CJ',
    'EXH': 'Exhaustive Prefixes (EXH)', 'HAM': 'Hamming Codes (HAM)',
    'RAND': 'Random (control)',
    'CRS': 'Contiguous Random (control)',
}

def load_predictions(pred_dir: str, asset: str) -> pd.DataFrame:
    path = os.path.join(pred_dir, f'{asset}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Predictions not found: {path}')
    df = pd.read_csv(path)
    # robust datetime
    date_col = 'Date' if 'Date' in df.columns else ('date' if 'date' in df.columns else None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
    return df

def compute_rolling_smape(df: pd.DataFrame, model: str, window: int) -> pd.Series:
    pred_col = f'Predicted_{model}'
    if pred_col not in df.columns:
        if 'Predicted' in df.columns:  # fallback if single model in CSV
            pred_col = 'Predicted'
        else:
            raise KeyError(f'Missing {pred_col} in predictions for model {model}')
    if 'Actual' not in df.columns:
        raise KeyError('Missing Actual column in predictions CSV')

    actual = df['Actual'].astype(float)
    pred = df[pred_col].astype(float)
    smape = 200.0 * (pred.sub(actual).abs()) / (pred.abs() + actual.abs() + EPS)
    roll = smape.rolling(window=window, min_periods=max(5, window // 4)).mean()
    return roll

def smooth_series(s: pd.Series, method: str = 'ema', span: int = 39) -> pd.Series:
    method = (method or 'ema').lower()
    if method == 'none':
        return s
    if method == 'ma':
        return s.rolling(window=max(3, span), min_periods=1).mean()
    # default: EMA
    return s.ewm(span=max(3, span), adjust=False).mean()

def add_linear_trend(ax, x_vals, y_vals, color, linewidth=1.2):
    x_num = np.arange(len(x_vals), dtype=float)
    y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 10:
        return
    coef = np.polyfit(x_num[mask], y[mask], 1)
    ax.plot(x_vals, np.polyval(coef, x_num), linestyle='--', linewidth=linewidth, color=color, alpha=0.9)

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
    # cleaner date ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha('center')

def apply_scale(ax, scale_str: str | None):
    if not scale_str:
        return
    try:
        lo, hi = [float(x) for x in scale_str.split(',')]
        ax.set_ylim(lo, hi)
    except Exception:
        pass

def prepare_style(style: str, n_lines: int):
    """Return colors and dash styles for up to 3 lines."""
    style = (style or 'monochrome').lower()
    if style == 'color':
        # Matplotlib paper-friendly palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        dashes = [(), (5,2.5), (3,2)]
    else:  # monochrome default
        colors = ['#222222', '#666666', '#999999']
        dashes = [(), (6,2.5), (3,2)]
    return colors[:n_lines], dashes[:n_lines]

def maybe_resample(s: pd.Series, resample_rule: str | None):
    if not resample_rule:
        return s
    # Take mean within the bin; drop NaNs produced by partial windows
    return s.resample(resample_rule).mean().dropna()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', required=True)
    ap.add_argument('--asset', required=True)
    ap.add_argument('--models', required=True, help='Comma-separated; choose 2â€“3 for readability')
    ap.add_argument('--window', type=int, default=78)
    ap.add_argument('--smooth', type=str, default='ema', choices=['none','ma','ema'])
    ap.add_argument('--smooth-span', type=int, default=39)
    ap.add_argument('--resample', type=str, default='W', help="Pandas resample rule (e.g., 'W' weekly, '2W', 'M'). Use '' to disable.")
    ap.add_argument('--trend', action='store_true', help='Overlay dashed linear fit per model')
    ap.add_argument('--scale', type=str, default=None, help="Clamp y-axis to 'MIN,MAX'")
    ap.add_argument('--style', type=str, default='monochrome', choices=['monochrome','color'])
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    if not (1 <= len(models) <= 3):
        raise SystemExit("Pick 1â€“3 models for a clean figure.")

    df = load_predictions(args.pred_dir, args.asset)

    # compute & smooth rolling SMAPE per model
    series = {}
    for m in models:
        r = compute_rolling_smape(df, m, window=args.window)
        r = smooth_series(r, method=args.smooth, span=args.smooth_span)
        r = maybe_resample(r, args.resample if args.resample else None)
        series[m] = r

    # align indexes
    idx = None
    for s in series.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    for m in series:
        series[m] = series[m].reindex(idx)

    # figure aesthetics
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
    })
    fig = plt.figure(figsize=(7.4, 3.25))
    ax = plt.gca()

    colors, dashes = prepare_style(args.style, len(models))

    for i, m in enumerate(models):
        label = DISPLAY_NAME.get(m, m)
        line, = ax.plot(idx, series[m].values, linewidth=1.55, label=label,
                        color=colors[i], zorder=2+i)
        line.set_dashes(dashes[i])
        if args.trend:
            add_linear_trend(ax, idx, series[m].values, color=colors[i], linewidth=0.9)

    title_asset = 'Cross-Sectional Average' if args.asset.upper() == 'ALL_MEAN' else args.asset
    ax.set_title(title_asset)
    ax.set_ylabel('Rolling SMAPE (%)')
    ax.set_xlabel('Time')
    style_axes(ax)
    apply_scale(ax, args.scale)
    ax.legend(frameon=False, ncol=min(3, len(models)), loc='upper right',
              borderaxespad=0.2, handlelength=2.4)

    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    pdf = os.path.join(args.outdir, f'Section_3_fig_2_temporal_stability_{args.asset}.pdf')
    svg = os.path.join(args.outdir, f'Section_3_fig_2_temporal_stability_{args.asset}.svg')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(svg, bbox_inches='tight')
    plt.close(fig)
    print(f'[ok] wrote {pdf}\n[ok] wrote {svg}')

if __name__ == '__main__':
    main()



