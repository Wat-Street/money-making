import os
import matplotlib.pyplot as plt

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

    plt.figure(figsize=(8, 5))
    plt.plot(corr_df["Lag"], corr_df["Correlation"], marker='o')
    plt.title(f"Lagged Correlation: {ticker_a} vs {ticker_b}")
    plt.xlabel("Lag (days)")
    plt.ylabel("Correlation")
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.grid(True)
    plt.tight_layout()

    if save:
        os.makedirs("outputs/plots", exist_ok=True)
        filename = f"outputs/plots/{ticker_a}_{ticker_b}_lagged_corr.png"
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        plt.close()
    else:
        plt.show()
