#!/usr/bin/env python3
"""
Compare Pearson vs Spearman correlation for all research pairs.
Generates comparison plots, summary statistics, and a results report.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from analysis.data_loader import load_pair_data
from analysis.pairs import research_pairs
from analysis.correlations import compute_lagged_correlation, summarize_best_lag
from analysis.plotting_compare import plot_lagged_correlation_compare, print_best_lag_summary

def compare_all_pairs(start_date="2020-01-01", end_date=None, max_lag=10, save_plots=True):
    """
    Compare Pearson vs Spearman correlations for all research pairs.
    
    Args:
        start_date: Start date for data
        end_date: End date for data (default: today)
        max_lag: Maximum lag to compute
        save_plots: Whether to save comparison plots
    
    Returns:
        pd.DataFrame: Summary results for all pairs
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    
    results = []
    os.makedirs("plots", exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Pearson vs Spearman Correlation Comparison")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Max lag: {max_lag}")
    print(f"Total pairs: {len(research_pairs)}")
    print(f"{'='*70}\n")
    
    for i, (ticker_a, ticker_b) in enumerate(research_pairs, 1):
        print(f"\n[{i}/{len(research_pairs)}] Processing: {ticker_a} vs {ticker_b}")
        print("-" * 70)
        
        try:
            # Load data
            df = load_pair_data(ticker_a, ticker_b, start_date, end_date, cache=True)
            if df.empty:
                print(f"❌ No data available")
                continue
            
            # Handle MultiIndex columns if present (fallback, data_loader should handle this)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0) if df.columns.nlevels > 1 else df.columns
            
            # Get the Close columns
            close_col_a = f"Close_{ticker_a}"
            close_col_b = f"Close_{ticker_b}"
            
            # Check if columns exist
            if close_col_a not in df.columns or close_col_b not in df.columns:
                print(f"Available columns: {df.columns.tolist()}")
                raise KeyError(f"Could not find Close columns: {close_col_a} or {close_col_b}")
            
            series_a = df[close_col_a]
            series_b = df[close_col_b]
            
            pearson_df = compute_lagged_correlation(series_a, series_b, max_lag=max_lag, method="pearson")
            spearman_df = compute_lagged_correlation(series_a, series_b, max_lag=max_lag, method="spearman")
            
            # Get best lag summaries
            p_summary = summarize_best_lag(pearson_df)
            s_summary = summarize_best_lag(spearman_df)
            
            # Calculate additional statistics
            pearson_max_abs = pearson_df["Correlation"].abs().max() if not pearson_df["Correlation"].isna().all() else None
            spearman_max_abs = spearman_df["Correlation"].abs().max() if not spearman_df["Correlation"].isna().all() else None
            
            pearson_at_zero = pearson_df[pearson_df["Lag"] == 0]["Correlation"].values[0] if len(pearson_df[pearson_df["Lag"] == 0]) > 0 else None
            spearman_at_zero = spearman_df[spearman_df["Lag"] == 0]["Correlation"].values[0] if len(spearman_df[spearman_df["Lag"] == 0]) > 0 else None
            
            # Difference metrics
            lag_diff = None
            corr_diff = None
            if p_summary["best_lag"] is not None and s_summary["best_lag"] is not None:
                lag_diff = abs(p_summary["best_lag"] - s_summary["best_lag"])
                corr_diff = abs(p_summary["best_corr"]) - abs(s_summary["best_corr"])
            
            zero_diff = None
            if pearson_at_zero is not None and spearman_at_zero is not None:
                zero_diff = abs(pearson_at_zero) - abs(spearman_at_zero)
            
            # Store results
            result = {
                "Ticker_A": ticker_a,
                "Ticker_B": ticker_b,
                "Pearson_Best_Lag": p_summary["best_lag"],
                "Pearson_Best_Corr": p_summary["best_corr"],
                "Pearson_Zero_Lag": pearson_at_zero,
                "Pearson_Max_Abs": pearson_max_abs,
                "Spearman_Best_Lag": s_summary["best_lag"],
                "Spearman_Best_Corr": s_summary["best_corr"],
                "Spearman_Zero_Lag": spearman_at_zero,
                "Spearman_Max_Abs": spearman_max_abs,
                "Lag_Difference": lag_diff,
                "Best_Corr_Difference": corr_diff,
                "Zero_Lag_Difference": zero_diff,
                "Data_Points": len(df),
            }
            results.append(result)
            
            # Print summary
            print_best_lag_summary(pearson_df, spearman_df)
            if pearson_at_zero is not None and spearman_at_zero is not None:
                print(f"Zero-lag: Pearson={pearson_at_zero:.4f}, Spearman={spearman_at_zero:.4f}")
            
            # Create comparison plot
            if save_plots:
                plot_lagged_correlation_compare(
                    ticker_a, ticker_b, df, max_lag=max_lag, save=True
                )
                print(f"✅ Saved comparison plot")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def generate_summary_report(results_df, output_file="results/pearson_spearman_comparison.csv"):
    """
    Generate a summary report from the results.
    
    Args:
        results_df: DataFrame with comparison results
        output_file: Path to save the CSV report
    """
    os.makedirs("results", exist_ok=True)
    
    # Save CSV
    results_df.to_csv(output_file, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nTotal pairs analyzed: {len(results_df)}")
    print(f"Pairs with valid Pearson results: {results_df['Pearson_Best_Corr'].notna().sum()}")
    print(f"Pairs with valid Spearman results: {results_df['Spearman_Best_Corr'].notna().sum()}")
    
    # Best correlation statistics
    print("\n--- Best Correlation (by absolute value) ---")
    if results_df['Pearson_Max_Abs'].notna().any():
        print(f"Pearson - Mean: {results_df['Pearson_Max_Abs'].mean():.4f}")
        print(f"Pearson - Median: {results_df['Pearson_Max_Abs'].median():.4f}")
        print(f"Pearson - Max: {results_df['Pearson_Max_Abs'].max():.4f}")
        print(f"Pearson - Min: {results_df['Pearson_Max_Abs'].min():.4f}")
    
    if results_df['Spearman_Max_Abs'].notna().any():
        print(f"Spearman - Mean: {results_df['Spearman_Max_Abs'].mean():.4f}")
        print(f"Spearman - Median: {results_df['Spearman_Max_Abs'].median():.4f}")
        print(f"Spearman - Max: {results_df['Spearman_Max_Abs'].max():.4f}")
        print(f"Spearman - Min: {results_df['Spearman_Max_Abs'].min():.4f}")
    
    # Zero-lag correlation statistics
    print("\n--- Zero-Lag Correlation ---")
    if results_df['Pearson_Zero_Lag'].notna().any():
        print(f"Pearson - Mean: {results_df['Pearson_Zero_Lag'].mean():.4f}")
        print(f"Pearson - Median: {results_df['Pearson_Zero_Lag'].median():.4f}")
    
    if results_df['Spearman_Zero_Lag'].notna().any():
        print(f"Spearman - Mean: {results_df['Spearman_Zero_Lag'].mean():.4f}")
        print(f"Spearman - Median: {results_df['Spearman_Zero_Lag'].median():.4f}")
    
    # Best lag statistics
    print("\n--- Best Lag Distribution ---")
    if results_df['Pearson_Best_Lag'].notna().any():
        print(f"Pearson best lag - Mean: {results_df['Pearson_Best_Lag'].mean():.2f}")
        print(f"Pearson best lag - Most common: {results_df['Pearson_Best_Lag'].mode().values[0] if len(results_df['Pearson_Best_Lag'].mode()) > 0 else 'N/A'}")
    
    if results_df['Spearman_Best_Lag'].notna().any():
        print(f"Spearman best lag - Mean: {results_df['Spearman_Best_Lag'].mean():.2f}")
        print(f"Spearman best lag - Most common: {results_df['Spearman_Best_Lag'].mode().values[0] if len(results_df['Spearman_Best_Lag'].mode()) > 0 else 'N/A'}")
    
    # Differences
    print("\n--- Method Differences ---")
    if results_df['Lag_Difference'].notna().any():
        print(f"Lag difference (absolute) - Mean: {results_df['Lag_Difference'].mean():.2f}")
        print(f"Lag difference (absolute) - Max: {results_df['Lag_Difference'].max():.0f}")
        print(f"Pairs with same best lag: {(results_df['Lag_Difference'] == 0).sum()}")
    
    if results_df['Best_Corr_Difference'].notna().any():
        print(f"Best correlation difference (Pearson - Spearman) - Mean: {results_df['Best_Corr_Difference'].mean():.4f}")
        print(f"Pairs where Pearson > Spearman: {(results_df['Best_Corr_Difference'] > 0).sum()}")
        print(f"Pairs where Spearman > Pearson: {(results_df['Best_Corr_Difference'] < 0).sum()}")
        print(f"Pairs where equal: {(results_df['Best_Corr_Difference'] == 0).sum()}")
    
    if results_df['Zero_Lag_Difference'].notna().any():
        print(f"Zero-lag difference (Pearson - Spearman) - Mean: {results_df['Zero_Lag_Difference'].mean():.4f}")
        print(f"Pairs where Pearson > Spearman at zero lag: {(results_df['Zero_Lag_Difference'] > 0).sum()}")
        print(f"Pairs where Spearman > Pearson at zero lag: {(results_df['Zero_Lag_Difference'] < 0).sum()}")
    
    # Top pairs by correlation
    print("\n--- Top 5 Pairs by Pearson Best Correlation ---")
    top_pearson = results_df.nlargest(5, 'Pearson_Max_Abs')[['Ticker_A', 'Ticker_B', 'Pearson_Best_Lag', 'Pearson_Best_Corr', 'Pearson_Max_Abs']]
    print(top_pearson.to_string(index=False))
    
    print("\n--- Top 5 Pairs by Spearman Best Correlation ---")
    top_spearman = results_df.nlargest(5, 'Spearman_Max_Abs')[['Ticker_A', 'Ticker_B', 'Spearman_Best_Lag', 'Spearman_Best_Corr', 'Spearman_Max_Abs']]
    print(top_spearman.to_string(index=False))
    
    print("\n" + "="*70)

def main():
    """Main function to run the comparison."""
    # Run comparison
    results_df = compare_all_pairs(
        start_date="2020-01-01",
        end_date=None,  # Will use today
        max_lag=10,
        save_plots=True
    )
    
    # Generate report
    if not results_df.empty:
        generate_summary_report(results_df)
    else:
        print("\n❌ No results to summarize")

if __name__ == "__main__":
    main()

