# compare_correlations.py
import argparse
import pandas as pd
from analysis.plotting_compare import (
    plot_lagged_correlation,
    plot_lagged_correlation_compare,
    print_best_lag_summary,
)

def main():
    parser = argparse.ArgumentParser(description="Compare Pearson vs Spearman lagged correlations.")
    parser.add_argument("--csv", required=True, help="Path to CSV with columns like Close_AAPL, Close_MSFT, ...")
    parser.add_argument("--ticker-a", required=True)
    parser.add_argument("--ticker-b", required=True)
    parser.add_argument("--max-lag", type=int, default=10)
    parser.add_argument("--save", action="store_true", help="Save plots to ./plots/")
    parser.add_argument("--method-only", choices=["pearson", "spearman"], help="If set, plot only this method.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.method_only:
        corr_df = plot_lagged_correlation(
            args.ticker_a, args.ticker_b, df, max_lag=args.max_lag, method=args.method_only, save=args.save
        )
        # Print table for quick inspection
        print(corr_df.to_string(index=False))
    else:
        pearson_df, spearman_df = plot_lagged_correlation_compare(
            args.ticker_a, args.ticker_b, df, max_lag=args.max_lag, save=args.save
        )
        print_best_lag_summary(pearson_df, spearman_df)

if __name__ == "__main__":
    main()
