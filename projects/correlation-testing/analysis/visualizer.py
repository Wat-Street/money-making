import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from analysis.correlation import compute_lagged_correlation

def plot_lagged_correlation(ticker_a, ticker_b, df, max_lag=10, save=False):
    """
    Plot lagged correlation between two stocks over a given lag range.

    Args:
        ticker_a (str): First stock ticker.
        ticker_b (str): Second stock ticker.
        df (pd.DataFrame): DataFrame with 'Close_<ticker>' columns.
        max_lag (int): Max lag to compute correlations over.
        save (bool): Whether to save plot to file.
    """
    series_a = df[f"Close_{ticker_a}"]
    series_b = df[f"Close_{ticker_b}"]
    corr_df = compute_lagged_correlation(series_a, series_b, max_lag)

    plt.figure(figsize=(10, 6))
    plt.plot(corr_df["Lag"], corr_df["Correlation"], marker='o', linewidth=2, markersize=6)
    plt.title(f"Lagged Correlation: {ticker_a} vs {ticker_b}")
    plt.xlabel("Lag (days)", fontsize=12)
    plt.ylabel("Correlation", fontsize=12)
    plt.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.grid(True)
    plt.tight_layout()

    if save:
        os.makedirs("plots", exist_ok=True)
        filename = f"plots/{ticker_a}_{ticker_b}_lagged_corr.png"
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        plt.close()
    
    plt.show()


def plot_spread_analysis(spread_df: pd.DataFrame, ticker_a: str, ticker_b: str, 
                        spread_type: str = 'log_ratio', save=False):
    """
    Create comprehensive spread analysis visualization with multiple subplots.
    
    Args:
        spread_df (pd.DataFrame): DataFrame from calculate_spread_metrics()
        ticker_a (str): First stock ticker
        ticker_b (str): Second stock ticker
        spread_type (str): Type of spread calculated
        save (bool): Whether to save plot to file
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f'Spread Analysis: {ticker_a} vs {ticker_b} ({spread_type})', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Normalized Price Series
    ax1 = axes[0]
    normalized_a = spread_df[f'{ticker_a}_price'] / spread_df[f'{ticker_a}_price'].iloc[0] * 100
    normalized_b = spread_df[f'{ticker_b}_price'] / spread_df[f'{ticker_b}_price'].iloc[0] * 100
    
    ax1.plot(spread_df['Date'], normalized_a, label=ticker_a, linewidth=2, alpha=0.8)
    ax1.plot(spread_df['Date'], normalized_b, label=ticker_b, linewidth=2, alpha=0.8)
    ax1.set_ylabel('Normalized Price (Base = 100)', fontsize=11)
    ax1.set_title('Normalized Price Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Spread with Mean and Standard Deviation Bands
    ax2 = axes[1]
    mean_val = spread_df['spread_mean'].iloc[0]
    std_val = spread_df['spread_std'].iloc[0]
    
    ax2.plot(spread_df['Date'], spread_df['spread'], label='Spread', 
            linewidth=1.5, color='navy', alpha=0.8)
    ax2.axhline(mean_val, color='red', linestyle='--', linewidth=2, label='Mean', alpha=0.8)
    ax2.axhline(mean_val + std_val, color='orange', linestyle=':', 
               linewidth=1.5, label='±1 Std', alpha=0.7)
    ax2.axhline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axhline(mean_val + 2*std_val, color='red', linestyle=':', 
               linewidth=1.5, label='±2 Std', alpha=0.7)
    ax2.axhline(mean_val - 2*std_val, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.fill_between(spread_df['Date'], mean_val - std_val, mean_val + std_val, 
                     alpha=0.2, color='orange')
    ax2.set_ylabel('Spread Value', fontsize=11)
    ax2.set_title('Spread with Mean Reversion Bands', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Z-Score
    ax3 = axes[2]
    ax3.plot(spread_df['Date'], spread_df['zscore'], label='Z-Score', 
            linewidth=1.5, color='darkgreen', alpha=0.8)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.axhline(2, color='red', linestyle='--', linewidth=1.5, label='Entry Threshold (±2)', alpha=0.7)
    ax3.axhline(-2, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.fill_between(spread_df['Date'], -2, 2, alpha=0.15, color='green', 
                     label='Normal Range')
    ax3.fill_between(spread_df['Date'], 2, spread_df['zscore'].max(), 
                     where=(spread_df['zscore'] > 2), alpha=0.2, color='red')
    ax3.fill_between(spread_df['Date'], spread_df['zscore'].min(), -2, 
                     where=(spread_df['zscore'] < -2), alpha=0.2, color='red')
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('Z-Score', fontsize=11)
    ax3.set_title('Spread Z-Score (Entry Signals)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        os.makedirs("plots/spreads", exist_ok=True)
        filename = f"plots/spreads/{ticker_a}_{ticker_b}_spread_analysis.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved spread analysis plot to {filename}")
        plt.close()
    else:
        plt.show()


def plot_spread_histogram(spread_df: pd.DataFrame, ticker_a: str, ticker_b: str, save=False):
    """
    Plot histogram of spread distribution to assess normality.
    
    Args:
        spread_df (pd.DataFrame): DataFrame with spread column
        ticker_a (str): First stock ticker
        ticker_b (str): Second stock ticker
        save (bool): Whether to save plot to file
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Spread Distribution Analysis: {ticker_a} vs {ticker_b}', 
                 fontsize=14, fontweight='bold')
    
    # Histogram of spread
    ax1 = axes[0]
    spread_values = spread_df['spread'].dropna()
    ax1.hist(spread_values, bins=50, alpha=0.7, color='navy', edgecolor='black')
    ax1.axvline(spread_values.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean = {spread_values.mean():.4f}')
    ax1.axvline(spread_values.median(), color='orange', linestyle='--', 
               linewidth=2, label=f'Median = {spread_values.median():.4f}')
    ax1.set_xlabel('Spread Value', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Spread Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of z-score
    ax2 = axes[1]
    zscore_values = spread_df['zscore'].dropna()
    ax2.hist(zscore_values, bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
    ax2.axvline(0, color='black', linestyle='-', linewidth=2, label='Mean = 0')
    ax2.axvline(2, color='red', linestyle='--', linewidth=1.5, label='±2 Std')
    ax2.axvline(-2, color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Z-Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Z-Score Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        os.makedirs("plots/spreads", exist_ok=True)
        filename = f"plots/spreads/{ticker_a}_{ticker_b}_spread_histogram.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved spread histogram to {filename}")
        plt.close()
    else:
        plt.show()


def plot_rolling_zscore_comparison(spread_df: pd.DataFrame, ticker_a: str, 
                                   ticker_b: str, save=False):
    """
    Compare z-scores calculated with different rolling windows.
    
    Args:
        spread_df (pd.DataFrame): DataFrame with multiple zscore columns
        ticker_a (str): First stock ticker
        ticker_b (str): Second stock ticker
        save (bool): Whether to save plot to file
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(spread_df['Date'], spread_df['zscore'], 
           label='Full History Z-Score', linewidth=2, alpha=0.8)
    
    if 'zscore_rolling_60d' in spread_df.columns:
        ax.plot(spread_df['Date'], spread_df['zscore_rolling_60d'], 
               label='60-Day Rolling Z-Score', linewidth=1.5, alpha=0.7)
    
    if 'zscore_rolling_252d' in spread_df.columns:
        ax.plot(spread_df['Date'], spread_df['zscore_rolling_252d'], 
               label='252-Day Rolling Z-Score', linewidth=1.5, alpha=0.7)
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Z-Score', fontsize=11)
    ax.set_title(f'Z-Score Comparison (Different Windows): {ticker_a} vs {ticker_b}', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        os.makedirs("plots/spreads", exist_ok=True)
        filename = f"plots/spreads/{ticker_a}_{ticker_b}_zscore_comparison.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved z-score comparison plot to {filename}")
        plt.close()
    else:
        plt.show()


def plot_trading_signals(spread_df: pd.DataFrame, ticker_a: str, ticker_b: str, save=False):
    """
    Visualize trading signals based on z-score thresholds.
    
    Args:
        spread_df (pd.DataFrame): DataFrame with signal columns from identify_trading_opportunities()
        ticker_a (str): First stock ticker
        ticker_b (str): Second stock ticker
        save (bool): Whether to save plot to file
    """
    if 'signal' not in spread_df.columns:
        print("Warning: No 'signal' column found. Run identify_trading_opportunities() first.")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Trading Signals: {ticker_a} vs {ticker_b}', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Spread with entry/exit points
    ax1 = axes[0]
    ax1.plot(spread_df['Date'], spread_df['spread'], 
            label='Spread', linewidth=1.5, color='navy', alpha=0.8)
    
    # Mark entry and exit points
    long_entries = spread_df[spread_df['signal'] == 'enter_long']
    short_entries = spread_df[spread_df['signal'] == 'enter_short']
    exits = spread_df[spread_df['signal'] == 'exit']
    
    ax1.scatter(long_entries['Date'], long_entries['spread'], 
               color='green', marker='^', s=100, label='Long Entry', zorder=5)
    ax1.scatter(short_entries['Date'], short_entries['spread'], 
               color='red', marker='v', s=100, label='Short Entry', zorder=5)
    ax1.scatter(exits['Date'], exits['spread'], 
               color='black', marker='x', s=100, label='Exit', zorder=5)
    
    mean_val = spread_df['spread_mean'].iloc[0]
    ax1.axhline(mean_val, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax1.set_ylabel('Spread Value', fontsize=11)
    ax1.set_title('Spread with Trading Signals', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Z-Score with threshold bands
    ax2 = axes[1]
    ax2.plot(spread_df['Date'], spread_df['zscore'], 
            label='Z-Score', linewidth=1.5, color='darkgreen', alpha=0.8)
    
    ax2.scatter(long_entries['Date'], long_entries['zscore'], 
               color='green', marker='^', s=100, label='Long Entry', zorder=5)
    ax2.scatter(short_entries['Date'], short_entries['zscore'], 
               color='red', marker='v', s=100, label='Short Entry', zorder=5)
    ax2.scatter(exits['Date'], exits['zscore'], 
               color='black', marker='x', s=100, label='Exit', zorder=5)
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axhline(2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.axhline(-2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.fill_between(spread_df['Date'], -2, 2, alpha=0.15, color='green')
    
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Z-Score', fontsize=11)
    ax2.set_title('Z-Score with Trading Signals', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        os.makedirs("plots/spreads", exist_ok=True)
        filename = f"plots/spreads/{ticker_a}_{ticker_b}_trading_signals.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved trading signals plot to {filename}")
        plt.close()
    else:
        plt.show()
