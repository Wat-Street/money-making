# plotting_compare.py
import os
import matplotlib.pyplot as plt
import pandas as pd
from analysis.correlations import compute_lagged_correlation, summarize_best_lag

def plot_lagged_correlation(
    ticker_a: str,
    ticker_b: str,
    df: pd.DataFrame,
    max_lag: int = 10,
    method: str = "pearson",
    save: bool = False,
) -> pd.DataFrame:
    """Plot a single-method (Pearson or Spearman) lagged correlation curve."""
    series_a = df[f"Close_{ticker_a}"]
    series_b = df[f"Close_{ticker_b}"]
    corr_df = compute_lagged_correlation(series_a, series_b, max_lag=max_lag, method=method)

    plt.figure(figsize=(10, 6))
    plt.plot(corr_df["Lag"], corr_df["Correlation"], marker="o", linewidth=2, markersize=6)
    plt.title(f"Lagged {method.capitalize()} Correlation: {ticker_a} vs {ticker_b}")
    plt.xlabel("Lag (days)")
    plt.ylabel("Correlation")
    plt.axhline(0, linestyle="--", linewidth=1, alpha=0.7)
    plt.axvline(0, linestyle="--", linewidth=1, alpha=0.7)
    plt.grid(True)
    plt.tight_layout()

    if save:
        os.makedirs("plots", exist_ok=True)
        filename = f"plots/{ticker_a}_{ticker_b}_lagged_{method.lower()}_corr.png"
        plt.savefig(filename, dpi=150)
        print(f"Saved plot to {filename}")
        plt.close()
    else:
        plt.show()
    return corr_df

def plot_lagged_correlation_compare(
    ticker_a: str,
    ticker_b: str,
    df: pd.DataFrame,
    max_lag: int = 10,
    save: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overlay Pearson and Spearman lagged correlations on the same figure."""
    series_a = df[f"Close_{ticker_a}"]
    series_b = df[f"Close_{ticker_b}"]

    pearson_df = compute_lagged_correlation(series_a, series_b, max_lag=max_lag, method="pearson")
    spearman_df = compute_lagged_correlation(series_a, series_b, max_lag=max_lag, method="spearman")

    plt.figure(figsize=(10, 6))
    plt.plot(pearson_df["Lag"], pearson_df["Correlation"], marker="o", linewidth=2, markersize=6, label="Pearson")
    plt.plot(spearman_df["Lag"], spearman_df["Correlation"], marker="s", linewidth=2, markersize=5, linestyle="--", label="Spearman")
    plt.title(f"Lagged Correlation Comparison: {ticker_a} vs {ticker_b}")
    plt.xlabel("Lag (days)")
    plt.ylabel("Correlation")
    plt.axhline(0, linestyle="--", linewidth=1, alpha=0.7)
    plt.axvline(0, linestyle="--", linewidth=1, alpha=0.7)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save:
        os.makedirs("plots", exist_ok=True)
        filename = f"plots/{ticker_a}_{ticker_b}_lagged_corr_compare.png"
        plt.savefig(filename, dpi=150)
        print(f"Saved plot to {filename}")
        plt.close()
    else:
        plt.show()
    return pearson_df, spearman_df

def print_best_lag_summary(pearson_df: pd.DataFrame, spearman_df: pd.DataFrame) -> None:
    p = summarize_best_lag(pearson_df)
    s = summarize_best_lag(spearman_df)
    print("=== Best lag summary (max |corr|) ===")
    print(f"Pearson : lag={p['best_lag']}, corr={p['best_corr']:.4f}" if p["best_lag"] is not None else "Pearson : no data")
    print(f"Spearman: lag={s['best_lag']}, corr={s['best_corr']:.4f}" if s["best_lag"] is not None else "Spearman: no data")
