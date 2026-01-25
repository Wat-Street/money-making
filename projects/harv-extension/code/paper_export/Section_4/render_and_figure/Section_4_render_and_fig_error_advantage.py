#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section 4 â€” "Where we win": Error Advantage vs HAR (Intraday focus)

Î”_t = loss(HAR)_t âˆ’ loss(Model)_t  (positive => Model better)
Filled area: green above zero (Model wins), red below zero (HAR wins)
One or two panels (you pick models), shared x-axis/y-axis.

Upgrades:
- --resample D/W/2W/M to aggregate deltas before plotting (default D)
- --smooth none|ma|ema with window/span (default ema, span 7)
- --clip-pctl L,H to clip plotted deltas to percentiles (e.g., 1,99)
- Stat box uses mathtext: Ã—10^{k} instead of 1.2e+05

Outputs:
  PDF + SVG figure, and a summary CSV with mean Î”, win rate, CLD, DM p-val.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

try:
    from scipy.stats import norm
    _has_scipy = True
except Exception:
    from math import erf, sqrt
    _has_scipy = False

EPS = 1e-12

DISPLAY_NAME = {
    'HAR': 'HAR-RV', 'HAR_J': 'HAR-RV-J', 'HAR_CJ': 'HAR-RV-CJ', 'HAR_TCJ': 'HAR-RV-TCJ',
    'PM': 'Prime Modulo (PM)', 'PM_VW': 'PM-VW', 'PM_AD': 'PM-AD',
    'CP': 'Contiguous Prime (CP)', 'CP_CJ': 'CP-CJ',
    'EXH': 'Exhaustive Prefixes (EXH)', 'HAM': 'Hamming Codes (HAM)',
    'RAND': 'Random (control)',
    'CRS': 'Contiguous Random (control)',
}

def _per_timestamp_smape_percent(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(EPS, np.abs(y_true) + np.abs(y_pred))
    return 200.0 * np.abs(y_true - y_pred) / denom

def _loss_series(y, yhat, metric: str):
    m = metric.lower()
    if m == 'mae':
        return np.abs(y - yhat)
    if m == 'smape':
        return _per_timestamp_smape_percent(y, yhat)
    raise ValueError("metric must be 'mae' or 'smape'")

def _dm_test(delta: np.ndarray, lag: int | None = None) -> float:
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 20:
        return np.nan
    d_bar = d.mean()
    if lag is None:
        lag = int(np.floor(1.5 * n ** (1/3)))  # common choice
    d_centered = d - d_bar
    gamma0 = np.dot(d_centered, d_centered) / n
    var_hat = gamma0
    for k in range(1, min(lag, n-1) + 1):
        w = 1.0 - k / (lag + 1.0)
        cov = np.dot(d_centered[:-k], d_centered[k:]) / n
        var_hat += 2.0 * w * cov
    var_mean = var_hat / n
    if var_mean <= 0 or not np.isfinite(var_mean):
        return np.nan
    t_stat = d_bar / np.sqrt(var_mean)
    if _has_scipy:
        p = 2.0 * (1.0 - norm.cdf(abs(t_stat)))
    else:
        p = 2.0 * (1.0 - 0.5 * (1 + erf(abs(t_stat) / np.sqrt(2))))
    return float(p)

def _sci_mathtext(x: float, sig=3) -> str:
    if not np.isfinite(x) or x == 0:
        return f"{x:.{sig}g}"
    e = int(np.floor(np.log10(abs(x))))
    m = x / (10 ** e)
    return rf"${m:.{sig}g}\times 10^{{{e}}}$"

def _pval_mathtext(p: float) -> str:
    if not np.isfinite(p):
        return "â€”"
    if p < 1e-6:
        return r"$<10^{-6}$"
    e = int(np.floor(np.log10(p)))
    m = p / (10 ** e)
    return rf"${m:.2g}\times 10^{{{e}}}$"

def _summarize(delta: np.ndarray) -> dict:
    pos = delta[delta > 0]
    neg = delta[delta < 0]
    return {
        'mean_delta': float(np.nanmean(delta)),
        'win_rate': float(np.mean(delta > 0)) if delta.size else np.nan,
        'cld': float(np.nansum(delta)),
        'median_delta': float(np.nanmedian(delta)),
        'avg_gain_when_better': float(np.nanmean(pos)) if pos.size else np.nan,
        'avg_loss_when_worse': float(-np.nanmean(neg)) if neg.size else np.nan,
        'p95_gain': float(np.nanpercentile(delta, 95)) if delta.size else np.nan,
        'p05_loss': float(np.nanpercentile(delta, 5)) if delta.size else np.nan,
        'dm_pvalue': _dm_test(delta),
    }

def _style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.6)

def _apply_scale(ax, scale: str | None):
    if not scale:
        return
    try:
        lo, hi = [float(x) for x in scale.split(',')]
        ax.set_ylim(lo, hi)
    except Exception:
        pass

def _smooth_series(s: pd.Series, method: str, span: int) -> pd.Series:
    method = (method or 'ema').lower()
    if method == 'none':
        return s
    if method == 'ma':
        return s.rolling(window=max(3, span), min_periods=1).mean()
    return s.ewm(span=max(3, span), adjust=False).mean()

def plot_error_advantage(
    df: pd.DataFrame, asset: str, baseline: str, models: list[str], metric: str,
    out_pdf: str, out_svg: str, out_csv: str,
    scale: str | None, alpha: float,
    resample_rule: str | None, smooth: str, smooth_span: int,
    clip_pctl: str | None,
):
    idx = df.index
    y = df['Actual'].values.astype(float)

    base_col = f'Predicted_{baseline}'
    if base_col not in df.columns:
        raise SystemExit(f"Baseline predictions not found: {base_col}")
    base_loss = _loss_series(y, df[base_col].values.astype(float), metric)

    n = len(models)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    green = '#2ca02c'
    red   = '#d62728'
    summary_rows = []

    for ax, m in zip(axes, models):
        col = f'Predicted_{m}'
        if col not in df.columns:
            raise SystemExit(f"Requested model not found: {col}")
        loss_m = _loss_series(y, df[col].values.astype(float), metric)
        delta_full = base_loss - loss_m  # stats use full-res
        stats = _summarize(delta_full)

        # For plotting: resample and smooth to avoid confetti
        s = pd.Series(delta_full, index=idx)
        if resample_rule:
            s = s.resample(resample_rule).median()
        if clip_pctl:
            try:
                lo_p, hi_p = [float(x) for x in clip_pctl.split(',')]
                lo_v, hi_v = np.nanpercentile(s.values, [lo_p, hi_p])
                s = s.clip(lower=lo_v, upper=hi_v)
            except Exception:
                pass
        s_smooth = _smooth_series(s, smooth, smooth_span)  # line overlay
        x = s.index
        yv = s.values

        ax.axhline(0.0, color='#333333', linewidth=0.9)
        ax.fill_between(x, 0, np.where(yv > 0, yv, 0), color=green, alpha=alpha, label=f'{DISPLAY_NAME.get(m,m)} better', interpolate=True)
        ax.fill_between(x, 0, np.where(yv < 0, yv, 0), color=red,   alpha=alpha*0.85, label='HAR-RV better', interpolate=True)
        ax.plot(x, s_smooth.values, color='#1a1a1a', linewidth=1.1, alpha=0.9)

        metric_lbl = 'SMAPE (%)' if metric.lower() == 'smape' else 'MAE'
        ax.set_ylabel(f'Î” {metric_lbl}')
        ax.set_title(f'Error Advantage vs {DISPLAY_NAME.get(baseline, baseline)} â€” {DISPLAY_NAME.get(m, m)}')
        _style_axes(ax)
        _apply_scale(ax, scale)
        ax.legend(loc='upper left', frameon=False, fontsize=8)

        # Stat box with mathtext
        mean_str = rf"Mean Î”: {stats['mean_delta']:.2f}" + (r"\%" if metric.lower() == 'smape' else "")
        win_str  = f"Win rate: {100*stats['win_rate']:.1f}%"
        cld_str  = fr"CLD: {_sci_mathtext(stats['cld'])}"
        p_str    = fr"DM p: {_pval_mathtext(stats['dm_pvalue'])}"
        box = AnchoredText("\n".join([mean_str, win_str, cld_str, p_str]),
                           loc='upper right', prop=dict(size=8),
                           frameon=True, borderpad=0.4)
        box.patch.set_alpha(0.90)
        ax.add_artist(box)

        summary_rows.append({
            'asset': asset, 'baseline': baseline, 'model': m, 'metric': metric.lower(),
            **stats
        })

    axes[-1].set_xlabel('Time')
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_svg, bbox_inches='tight')
    plt.close(fig)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', required=True)
    ap.add_argument('--asset', required=True)
    ap.add_argument('--baseline', default='HAR')
    ap.add_argument('--models', required=True, help='Comma-separated list, e.g., PM,CP')
    ap.add_argument('--metric', default='smape', choices=['smape','mae'])
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--prefix', default='Section_4_fig_3_error_advantage')
    ap.add_argument('--start', type=str, default=None)
    ap.add_argument('--end', type=str, default=None)
    ap.add_argument('--scale', type=str, default=None, help="Clamp y-axis to 'MIN,MAX'")
    ap.add_argument('--alpha', type=float, default=0.30)
    ap.add_argument('--resample', type=str, default='D', help="Pandas rule: D (default), W, 2W, M. Use '' to disable.")
    ap.add_argument('--smooth', type=str, default='ema', choices=['none','ma','ema'])
    ap.add_argument('--smooth-span', type=int, default=7)
    ap.add_argument('--clip-pctl', type=str, default='1,99', help="Clip plotted deltas to percentiles L,H (e.g., 1,99). Use '' to disable.")
    args = ap.parse_args()

    csv_path = os.path.join(args.pred_dir, f'{args.asset}.csv')
    if not os.path.exists(csv_path):
        raise SystemExit(f"Predictions CSV not found for {args.asset}: {csv_path}")
    df = pd.read_csv(csv_path)
    dt_col = 'Date' if 'Date' in df.columns else ('date' if 'date' in df.columns else None)
    if dt_col is None:
        raise SystemExit("Predictions CSV must contain a Date/date column.")
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.set_index(dt_col).sort_index()

    # date filters
    if args.start:
        df = df[df.index >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df.index <= pd.to_datetime(args.end)]
    if df.empty:
        raise SystemExit("No data after date filtering.")

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    if not (1 <= len(models) <= 2):
        raise SystemExit("Pick 1â€“2 models for a clean figure.")

    out_pdf = os.path.join(args.outdir, f"{args.prefix}_{args.asset}.pdf")
    out_svg = os.path.join(args.outdir, f"{args.prefix}_{args.asset}.svg")
    out_csv = os.path.join("code/outputs/intraday/tables", f"Section_4_error_advantage_summary_{args.asset}.csv")

    plot_error_advantage(
        df=df,
        asset=args.asset,
        baseline=args.baseline,
        models=models,
        metric=args.metric,
        out_pdf=out_pdf,
        out_svg=out_svg,
        out_csv=out_csv,
        scale=args.scale,
        alpha=args.alpha,
        resample_rule=(args.resample if args.resample else None),
        smooth=args.smooth,
        smooth_span=args.smooth_span,
        clip_pctl=(args.clip_pctl if args.clip_pctl else None),
    )
    print(f"[ok] wrote {out_pdf}\n[ok] wrote {out_svg}\n[ok] wrote {out_csv}")

if __name__ == '__main__':
    main()



