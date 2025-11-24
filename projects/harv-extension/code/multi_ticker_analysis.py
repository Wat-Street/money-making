#!/usr/bin/env python3
"""
Multi-Ticker Analysis Script
- Runs models on multiple tickers and generates comparative visualizations
- Allows for side-by-side comparison of model performance across different stocks
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

# For parallel processing
from concurrent.futures import ProcessPoolExecutor
import subprocess
import sys

# Constants
EPS = 1e-12
DISPLAY_NAME = {
    'HAR': 'HAR-RV', 'HAR_J': 'HAR-RV-J', 'HAR_CJ': 'HAR-RV-CJ', 'HAR_TCJ': 'HAR-RV-TCJ',
    'PM': 'Prime Modulo (PM)', 'PM_VW': 'PM-VW', 'PM_AD': 'PM-AD',
    'CP': 'Contiguous Prime (CP)', 'CP_CJ': 'CP-CJ',
    'EXH': 'Exhaustive Prefixes (EXH)', 'HAM': 'Hamming Codes (HAM)',
    'RAND': 'Random (control)', 'CRS': 'Contiguous Random Sets (CRS)',
}

def list_available_tickers(data_dir):
    """
    List all available ticker datasets in the specified directory.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing ticker data files (e.g., 'code/Datasets/clean')
        
    Returns:
    --------
    list
        List of available ticker symbols (without the '_5m.csv' suffix)
    """
    if not os.path.isdir(data_dir):
        print(f"Warning: Directory {data_dir} does not exist")
        return []
    
    tickers = []
    for filename in os.listdir(data_dir):
        if filename.endswith('_5m.csv'):
            ticker = filename.replace('_5m.csv', '')
            tickers.append(ticker)
    
    return sorted(tickers)

def load_predictions(pred_dir: str, asset: str) -> pd.DataFrame:
    """Load prediction data for a specific asset."""
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
    """Compute rolling SMAPE for a specific model."""
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
    """Apply smoothing to a time series."""
    method = (method or 'ema').lower()
    if method == 'none':
        return s
    if method == 'ma':
        return s.rolling(window=max(3, span), min_periods=1).mean()
    # default: EMA
    return s.ewm(span=max(3, span), adjust=False).mean()

def maybe_resample(s: pd.Series, resample_rule: str | None):
    """Resample a time series using the specified rule."""
    if not resample_rule:
        return s
    # Take mean within the bin; drop NaNs produced by partial windows
    return s.resample(resample_rule).mean().dropna()

def prepare_style(style: str, n_lines: int):
    """Return colors and dash styles for up to n_lines."""
    style = (style or 'monochrome').lower()
    if style == 'color':
        # Matplotlib paper-friendly palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        dashes = [(), (5,2.5), (3,2), (1,1), (5,1,2,1), (3,1,1,1)]
    else:  # monochrome default
        colors = ['#000000', '#333333', '#666666', '#999999', '#bbbbbb', '#dddddd']
        dashes = [(), (6,2.5), (3,2), (1,1), (5,1,2,1), (3,1,1,1)]
    
    # Extend patterns if needed
    while len(colors) < n_lines:
        colors.extend(colors[:n_lines-len(colors)])
    while len(dashes) < n_lines:
        dashes.extend(dashes[:n_lines-len(dashes)])
        
    return colors[:n_lines], dashes[:n_lines]

def style_axes(ax):
    """Apply consistent styling to plot axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.6)
    # cleaner date ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha('center')

def apply_scale(ax, scale_str: str | None):
    """Apply y-axis scale limits."""
    if not scale_str:
        return
    try:
        lo, hi = [float(x) for x in scale_str.split(',')]
        ax.set_ylim(lo, hi)
    except Exception:
        pass

def run_intraday_benchmark(ticker, models, n, warmup, local_dir, outdir):
    """Run the intraday benchmark for a single ticker."""
    cmd = [
        sys.executable, 'code/intraday_benchmark.py',
        '--tickers', ticker,
        '--models', ','.join(models),
        '--n', str(n),
        '--warmup', str(warmup),
        '--local-dir', local_dir,
        '--outdir', outdir
    ]
    subprocess.run(cmd, check=True)

def run_models_on_multiple_tickers(tickers, models, n, warmup, local_dir, outdir, parallel=False):
    """Run models on multiple tickers, optionally in parallel."""
    os.makedirs(os.path.join(outdir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'tables'), exist_ok=True)
    
    if parallel:
        with ProcessPoolExecutor() as executor:
            futures = []
            for ticker in tickers:
                future = executor.submit(
                    run_intraday_benchmark,
                    ticker, models, n, warmup, local_dir, outdir
                )
                futures.append(future)
            
            for i, future in enumerate(futures):
                try:
                    future.result()
                    print(f"[DONE] {tickers[i]}")
                except Exception as e:
                    print(f"[ERROR] {tickers[i]}: {e}")
    else:
        for ticker in tickers:
            try:
                run_intraday_benchmark(ticker, models, n, warmup, local_dir, outdir)
                print(f"[DONE] {ticker}")
            except Exception as e:
                print(f"[ERROR] {ticker}: {e}")

def plot_multi_ticker_temporal_stability(pred_dir, tickers, models, window=78, 
                                        smooth='ema', smooth_span=39, resample='W',
                                        trend=False, scale=None, style='color', outdir='code/paper/figures'):
    """
    Generate a multi-ticker temporal stability plot.
    
    Parameters:
    -----------
    pred_dir : str
        Directory containing prediction CSV files
    tickers : list
        List of ticker symbols to include
    models : list
        List of model names to include
    window : int
        Rolling window size for SMAPE calculation
    smooth : str
        Smoothing method ('none', 'ma', or 'ema')
    smooth_span : int
        Span parameter for smoothing
    resample : str
        Pandas resample rule (e.g., 'W' for weekly)
    trend : bool
        Whether to overlay linear trend lines
    scale : str
        Y-axis scale limits in format 'min,max'
    style : str
        Plot style ('monochrome' or 'color')
    outdir : str
        Output directory for figures
    """
    # Set up the figure
    n_tickers = len(tickers)
    fig_height = 4 * n_tickers
    fig_width = 12
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(n_tickers, 1, figure=fig)
    
    # Process each ticker
    for i, ticker in enumerate(tickers):
        try:
            # Load data
            df = load_predictions(pred_dir, ticker)
            
            # Compute & smooth rolling SMAPE per model
            series = {}
            for m in models:
                try:
                    r = compute_rolling_smape(df, m, window=window)
                    r = smooth_series(r, method=smooth, span=smooth_span)
                    r = maybe_resample(r, resample if resample else None)
                    series[m] = r
                except Exception as e:
                    print(f"Warning: Could not process {m} for {ticker}: {e}")
            
            if not series:
                print(f"No valid models for {ticker}, skipping...")
                continue
                
            # Align indexes
            idx = None
            for s in series.values():
                idx = s.index if idx is None else idx.intersection(s.index)
            for m in series:
                series[m] = series[m].reindex(idx)
            
            # Create subplot
            ax = fig.add_subplot(gs[i, 0])
            
            colors, dashes = prepare_style(style, len(models))
            
            for j, m in enumerate(models):
                if m not in series:
                    continue
                    
                label = f"{DISPLAY_NAME.get(m, m)}"
                line, = ax.plot(idx, series[m].values, linewidth=1.6, label=label,
                              color=colors[j], zorder=2+j)
                line.set_dashes(dashes[j])
                
                if trend:
                    # Add linear trend
                    x_num = np.arange(len(idx), dtype=float)
                    y = np.asarray(series[m].values, dtype=float)
                    mask = np.isfinite(y)
                    if mask.sum() >= 10:
                        coef = np.polyfit(x_num[mask], y[mask], 1)
                        ax.plot(idx, np.polyval(coef, x_num), linestyle='--', 
                              linewidth=1.1, color=colors[j], alpha=0.9)
            
            # Style the subplot
            ax.set_title(f'Rolling SMAPE (window={window}) — {ticker}')
            ax.set_ylabel('Rolling SMAPE (%)')
            if i == n_tickers - 1:  # Only add x-label to bottom subplot
                ax.set_xlabel('Time')
            style_axes(ax)
            apply_scale(ax, scale)
            ax.legend(frameon=False, ncol=min(3, len(models)), loc='upper left')
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    # Finalize the figure
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, f'Multi_Ticker_Temporal_Stability.pdf')
    svg = os.path.join(outdir, f'Multi_Ticker_Temporal_Stability.svg')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(svg, bbox_inches='tight')
    print(f'[ok] wrote {pdf}\n[ok] wrote {svg}')
    return fig

def plot_model_comparison_across_tickers(pred_dir, tickers, models, metric='SMAPE', outdir='code/paper/figures'):
    """
    Generate a bar chart comparing model performance across multiple tickers.
    
    Parameters:
    -----------
    pred_dir : str
        Directory containing prediction CSV files
    tickers : list
        List of ticker symbols to include
    models : list
        List of model names to include
    metric : str
        Performance metric to compare ('SMAPE' or 'AbsErr')
    outdir : str
        Output directory for figures
    """
    # Collect performance metrics for each ticker and model
    results = []
    
    for ticker in tickers:
        try:
            df = load_predictions(pred_dir, ticker)
            
            for model in models:
                metric_col = f'{metric}_{model}_pct' if metric == 'SMAPE' else f'{metric}_{model}'
                
                if metric_col not in df.columns:
                    # Try alternative column name formats
                    alt_cols = [f'{model}_{metric}_pct', f'{model}_{metric}']
                    found = False
                    for col in alt_cols:
                        if col in df.columns:
                            metric_col = col
                            found = True
                            break
                    
                    if not found:
                        print(f"Warning: Metric {metric} not found for {model} in {ticker}")
                        continue
                
                mean_value = df[metric_col].mean()
                std_value = df[metric_col].std()
                
                results.append({
                    'Ticker': ticker,
                    'Model': DISPLAY_NAME.get(model, model),
                    'Mean': mean_value,
                    'Std': std_value
                })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    if not results:
        print("No valid results to plot")
        return None
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Use seaborn for a nicer grouped bar chart
    ax = sns.barplot(x='Ticker', y='Mean', hue='Model', data=results_df, errorbar=('ci', 95))
    
    # Style the plot
    ax.set_title(f'Model Comparison Across Tickers ({metric})')
    ax.set_ylabel(f'Mean {metric} (%)' if metric == 'SMAPE' else f'Mean {metric}')
    ax.set_xlabel('Ticker')
    
    # Rotate x-tick labels if there are many tickers
    if len(tickers) > 5:
        plt.xticks(rotation=45)
    
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, f'Model_Comparison_Across_Tickers_{metric}.pdf')
    svg = os.path.join(outdir, f'Model_Comparison_Across_Tickers_{metric}.svg')
    plt.savefig(pdf, bbox_inches='tight')
    plt.savefig(svg, bbox_inches='tight')
    print(f'[ok] wrote {pdf}\n[ok] wrote {svg}')
    
    return plt.gcf()

def plot_heatmap_model_performance(pred_dir, tickers, models, metric='SMAPE', outdir='code/paper/figures'):
    """
    Generate a heatmap of model performance across multiple tickers.
    
    Parameters:
    -----------
    pred_dir : str
        Directory containing prediction CSV files
    tickers : list
        List of ticker symbols to include
    models : list
        List of model names to include
    metric : str
        Performance metric to compare ('SMAPE' or 'AbsErr')
    outdir : str
        Output directory for figures
    """
    # Collect performance metrics for each ticker and model
    results = {}
    
    for ticker in tickers:
        results[ticker] = {}
        try:
            df = load_predictions(pred_dir, ticker)
            
            for model in models:
                metric_col = f'{metric}_{model}_pct' if metric == 'SMAPE' else f'{metric}_{model}'
                
                if metric_col not in df.columns:
                    # Try alternative column name formats
                    alt_cols = [f'{model}_{metric}_pct', f'{model}_{metric}']
                    found = False
                    for col in alt_cols:
                        if col in df.columns:
                            metric_col = col
                            found = True
                            break
                    
                    if not found:
                        print(f"Warning: Metric {metric} not found for {model} in {ticker}")
                        results[ticker][DISPLAY_NAME.get(model, model)] = np.nan
                        continue
                
                mean_value = df[metric_col].mean()
                results[ticker][DISPLAY_NAME.get(model, model)] = mean_value
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            for model in models:
                results[ticker][DISPLAY_NAME.get(model, model)] = np.nan
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results).T  # Transpose so tickers are rows
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    
    # Normalize for better color mapping
    norm_df = results_df.copy()
    for col in norm_df.columns:
        col_max = norm_df[col].max()
        col_min = norm_df[col].min()
        if col_max > col_min:
            norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
    
    # Plot heatmap
    ax = sns.heatmap(norm_df, annot=results_df.round(2), fmt='.2f', cmap='YlGnBu_r',
                    linewidths=0.5, cbar_kws={'label': f'Normalized {metric}'})
    
    # Style the plot
    ax.set_title(f'Model Performance Across Tickers ({metric})')
    ax.set_xlabel('Model')
    ax.set_ylabel('Ticker')
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, f'Model_Performance_Heatmap_{metric}.pdf')
    svg = os.path.join(outdir, f'Model_Performance_Heatmap_{metric}.svg')
    plt.savefig(pdf, bbox_inches='tight')
    plt.savefig(svg, bbox_inches='tight')
    print(f'[ok] wrote {pdf}\n[ok] wrote {svg}')
    
    return plt.gcf()

def main():
    ap = argparse.ArgumentParser(description='Multi-Ticker Analysis Tool')
    ap.add_argument('--tickers', type=str, help='Comma-separated ticker symbols')
    ap.add_argument('--models', type=str, default='HAR,PM,CP', help='Comma-separated model names')
    ap.add_argument('--local-dir', type=str, default='code/Datasets/clean', help='Directory with local data files')
    ap.add_argument('--outdir', type=str, default='code/outputs/multi_ticker', help='Output directory')
    ap.add_argument('--pred-dir', type=str, help='Directory with prediction files (if already generated)')
    ap.add_argument('--n', type=int, default=22, help='Prediction horizon')
    ap.add_argument('--warmup', type=int, default=600, help='Warmup period')
    ap.add_argument('--run-models', action='store_true', help='Run models on tickers (otherwise just generate plots)')
    ap.add_argument('--parallel', action='store_true', help='Run models in parallel')
    ap.add_argument('--list-tickers', action='store_true', help='List available tickers and exit')
    
    # Plotting parameters
    ap.add_argument('--window', type=int, default=78, help='Rolling window size for SMAPE')
    ap.add_argument('--smooth', type=str, default='ema', choices=['none','ma','ema'], help='Smoothing method')
    ap.add_argument('--smooth-span', type=int, default=39, help='Smoothing span')
    ap.add_argument('--resample', type=str, default='W', help='Resampling rule')
    ap.add_argument('--trend', action='store_true', help='Add trend lines')
    ap.add_argument('--scale', type=str, help='Y-axis scale (min,max)')
    ap.add_argument('--style', type=str, default='color', choices=['monochrome','color'], help='Plot style')
    ap.add_argument('--fig-outdir', type=str, default='code/paper/figures', help='Figure output directory')
    
    args = ap.parse_args()
    
    # List available tickers if requested
    available_tickers = list_available_tickers(args.local_dir)
    if args.list_tickers:
        print("Available tickers:")
        for ticker in available_tickers:
            print(f"  {ticker}")
        return
    
    # If no tickers specified, use all available ones
    if not args.tickers:
        if not available_tickers:
            print(f"Error: No ticker data files found in {args.local_dir}")
            return
        tickers = available_tickers
        print(f"No tickers specified, using all {len(tickers)} available tickers")
    else:
        # Parse tickers and verify they exist
        requested_tickers = [t.strip() for t in args.tickers.split(',') if t.strip()]
        tickers = []
        for ticker in requested_tickers:
            if ticker in available_tickers:
                tickers.append(ticker)
            else:
                print(f"Warning: No data found for ticker {ticker}, skipping")
        
        if not tickers:
            print("Error: None of the requested tickers have data files")
            print(f"Available tickers: {', '.join(available_tickers)}")
            return
    
    # Parse models
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    
    print(f"Processing {len(tickers)} tickers: {', '.join(tickers)}")
    print(f"Using {len(models)} models: {', '.join(models)}")
    
    # Run models if requested
    if args.run_models:
        print(f"Running models on tickers...")
        run_models_on_multiple_tickers(
            tickers, models, args.n, args.warmup, args.local_dir, args.outdir, args.parallel
        )
    
    # Determine prediction directory
    pred_dir = args.pred_dir if args.pred_dir else os.path.join(args.outdir, 'predictions')
    
    # Generate plots
    print("Generating temporal stability plot...")
    plot_multi_ticker_temporal_stability(
        pred_dir, tickers, models, args.window, args.smooth, args.smooth_span,
        args.resample, args.trend, args.scale, args.style, args.fig_outdir
    )
    
    print("Generating model comparison plot...")
    plot_model_comparison_across_tickers(
        pred_dir, tickers, models, 'SMAPE', args.fig_outdir
    )
    
    print("Generating performance heatmap...")
    plot_heatmap_model_performance(
        pred_dir, tickers, models, 'SMAPE', args.fig_outdir
    )
    
    print("All done!")

if __name__ == '__main__':
    main()
