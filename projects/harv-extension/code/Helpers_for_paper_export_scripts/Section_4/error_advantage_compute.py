import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This script is standalone; it reads the predictions CSVs written by your runners.
# It requires explicit model and asset selection; there is no auto best-of logic.

def _per_timestamp_smape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(eps, np.abs(y_true) + np.abs(y_pred))
    return 200.0 * np.abs(y_true - y_pred) / denom

def _loss_series(y: np.ndarray, yhat: np.ndarray, metric: str) -> np.ndarray:
    if metric.lower() == 'mae':
        return np.abs(y - yhat)
    elif metric.lower() == 'smape':
        return _per_timestamp_smape_percent(y, yhat)
    else:
        raise ValueError("metric must be one of: 'mae', 'smape'")

def _summarize_delta(delta: np.ndarray) -> dict:
    # delta = loss_baseline - loss_model  (positive => model better)
    pos = delta[delta > 0]
    neg = delta[delta < 0]
    return {
        'outperform_frac': float(np.mean(delta > 0)) if delta.size else np.nan,
        'avg_gain_when_better': float(np.mean(pos)) if pos.size else np.nan,
        'avg_loss_when_worse': float(-np.mean(neg)) if neg.size else np.nan,  # positive magnitude
        'net_avg_delta': float(np.mean(delta)) if delta.size else np.nan,
        'median_delta': float(np.median(delta)) if delta.size else np.nan,
        'p95_gain': float(np.nanpercentile(delta, 95)) if delta.size else np.nan,
        'p05_loss': float(np.nanpercentile(delta, 5)) if delta.size else np.nan,
    }

def plot_error_advantage(
    df: pd.DataFrame,
    baseline: str,
    models: list,
    metric: str,
    outfile_pdf: str,
    stacked: bool = True
):
    idx = df.index
    y = df['Actual'].values.astype(float)

    # Baseline loss
    base_col = f'Predicted_{baseline}'
    if base_col not in df.columns:
        raise SystemExit(f"Baseline predictions not found: {base_col}")
    yb = df[base_col].values.astype(float)
    base_loss = _loss_series(y, yb, metric)

    # Prepare figure
    if stacked and len(models) > 1:
        nrows = len(models)
        fig, axes = plt.subplots(nrows, 1, figsize=(12, 2.8 * nrows), sharex=True)
        if nrows == 1:
            axes = [axes]
    else:
        fig, ax = plt.subplots(figsize=(12, 4.2))
        axes = [ax]

    # Plot per model
    for i, m in enumerate(models):
        col = f'Predicted_{m}'
        if col not in df.columns:
            raise SystemExit(f"Requested model not found in predictions: {col}")

        ym = df[col].values.astype(float)
        model_loss = _loss_series(y, ym, metric)
        delta = base_loss - model_loss  # >0 means model beats baseline

        ax = axes[i if (stacked and len(models) > 1) else 0]
        ax.plot(idx, np.zeros_like(delta), linewidth=0.8)
        ax.fill_between(idx, 0, delta, where=(delta >= 0), alpha=0.35, label=f'{m} better', interpolate=True)
        ax.fill_between(idx, 0, delta, where=(delta < 0), alpha=0.25, label=f'{m} worse', interpolate=True)
        ax.set_ylabel(f'Δ {metric.upper()} (BASE−{m})')
        ax.set_title(f'Error Advantage vs {baseline} — {m}')

        # Optional: thin grid
        ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.6)

        # Keep legend compact
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', ncol=2, fontsize=8)

    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile_pdf), exist_ok=True)
    plt.savefig(outfile_pdf, bbox_inches='tight')
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Error Advantage vs Baseline (area chart + summary)")
    ap.add_argument('--pred-dir', type=str, required=True, help='Directory with per-asset predictions CSVs')
    ap.add_argument('--asset', type=str, required=True, help='Ticker to analyze (e.g., SPY)')
    ap.add_argument('--baseline', type=str, default='HAR', help='Baseline model code (default HAR)')
    ap.add_argument('--models', type=str, required=True, help='Comma-separated comparison models (e.g., PM,CP)')
    ap.add_argument('--metric', type=str, default='mae', choices=['mae','smape'], help='Loss metric for advantage')
    ap.add_argument('--out-fig', type=str, required=True, help='Output PDF path for figure')
    ap.add_argument('--out-csv', type=str, required=True, help='Output CSV path for per-model advantage summary')
    ap.add_argument('--start', type=str, default=None, help='Optional start date filter YYYY-MM-DD')
    ap.add_argument('--end', type=str, default=None, help='Optional end date filter YYYY-MM-DD')
    ap.add_argument('--stacked', action='store_true', help='Use stacked subplots if multiple models')
    args = ap.parse_args()

    asset_csv = os.path.join(args.pred_dir, f'{args.asset}.csv')
    if not os.path.exists(asset_csv):
        raise SystemExit(f"Predictions CSV not found for asset {args.asset}: {asset_csv}")

    df = pd.read_csv(asset_csv, parse_dates=['Date']).set_index('Date')
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    if not models:
        raise SystemExit("Provide at least one model in --models.")
    if f'Predicted_{args.baseline}' not in df.columns:
        raise SystemExit(f"Baseline not found in file: Predicted_{args.baseline}")

    # date filter
    if args.start:
        df = df[df.index >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df.index <= pd.to_datetime(args.end)]
    if df.empty:
        raise SystemExit("No data after date filtering.")

    # Validate requested models all exist
    missing = [m for m in models if f'Predicted_{m}' not in df.columns]
    if missing:
        raise SystemExit(f"Requested models not found in file: {missing}")

    # Compute and save figure
    plot_error_advantage(df, args.baseline, models, args.metric, args.out_fig, stacked=args.stacked)

    # Build per-model summary CSV
    y = df['Actual'].values.astype(float)
    base_loss = _loss_series(y, df[f'Predicted_{args.baseline}'].values.astype(float), args.metric)
    rows = []
    for m in models:
        model_loss = _loss_series(y, df[f'Predicted_{m}'].values.astype(float), args.metric)
        delta = base_loss - model_loss
        summ = _summarize_delta(delta)
        rows.append({'asset': args.asset, 'baseline': args.baseline, 'model': m, 'metric': args.metric, **summ})

    out_dir = os.path.dirname(args.out_csv)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_fig}")
    print(f"Wrote {args.out_csv}")

if __name__ == '__main__':
    main()
