from datetime import datetime
from analysis.data_loader import load_pair_data
from analysis.spread import calculate_spread_metrics
from analysis.visualizer import plot_spread_analysis, plot_spread_histogram
from analysis.pairs import research_pairs


def analyze_pair(ticker_a, ticker_b, start_date, end_date, spread_type='log_ratio'):
    """Analyze and visualize a single pair."""
    
    print(f"\n{'='*60}")
    print(f"ANALYZING: {ticker_a} vs {ticker_b}")
    print(f"{'='*60}")
    
    try:
        # Load data
        print(f"\nLoading data from {start_date} to {end_date}...")
        df = load_pair_data(ticker_a, ticker_b, start_date, end_date, cache=True)
        
        if df.empty:
            print(f"❌ No data available for {ticker_a} and {ticker_b}")
            return False
        
        print(f"✅ Loaded {len(df)} data points")
        
        # Calculate spread
        print(f"Calculating {spread_type} spread...")
        spread_df, metrics = calculate_spread_metrics(df, ticker_a, ticker_b, spread_type=spread_type)
        
        # Add extra columns for visualization
        spread_df['spread_mean'] = metrics['mean']
        spread_df['spread_std'] = metrics['std']
        
        print(f"\n Spread Metrics:")
        print(f"   Mean:            {metrics['mean']:.6f}")
        print(f"   Std Dev:         {metrics['std']:.6f}")
        print(f"   Current:         {metrics['current']:.6f}")
        print(f"   Current Z-Score: {metrics['current_zscore']:.2f}")
        
        # Trading signal
        if metrics['current_zscore'] > 2:
            print(f"   💡 Signal: SHORT spread (sell {ticker_a}, buy {ticker_b})")
        elif metrics['current_zscore'] < -2:
            print(f"   💡 Signal: LONG spread (buy {ticker_a}, sell {ticker_b})")
        else:
            print(f"   💡 Signal: HOLD (no trade)")
        
        # Generate visualizations
        print(f"\n📈 Generating visualizations...")
        plot_spread_analysis(spread_df, ticker_a, ticker_b, spread_type=spread_type, save=True)
        print(f"   ✅ Saved spread analysis")
        
        plot_spread_histogram(spread_df, ticker_a, ticker_b, save=True)
        print(f"   ✅ Saved histogram")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing {ticker_a}-{ticker_b}: {e}")
        return False


def main():
    """Run spread analysis on research pairs."""
    
    start_date = "2023-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    spread_type = 'log_ratio'  # Options: 'difference', 'ratio', 'log_ratio'
    
    print(f"\n{'#'*60}")
    print(f"# SPREAD ANALYSIS FOR RESEARCH PAIRS")
    print(f"# Period: {start_date} to {end_date}")
    print(f"# Spread Type: {spread_type}")
    print(f"# Total Pairs: {len(research_pairs)}")
    print(f"{'#'*60}")
    
    # Process each pair
    success_count = 0
    failed_pairs = []
    
    for i, (ticker_a, ticker_b) in enumerate(research_pairs, 1):
        print(f"\n[{i}/{len(research_pairs)}]", end=" ")
        
        if analyze_pair(ticker_a, ticker_b, start_date, end_date, spread_type):
            success_count += 1
        else:
            failed_pairs.append((ticker_a, ticker_b))
    
    # Summary
    print(f"\n\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully analyzed: {success_count}/{len(research_pairs)} pairs")
    
    if failed_pairs:
        print(f"\n❌ Failed pairs:")
        for ticker_a, ticker_b in failed_pairs:
            print(f"   - {ticker_a}/{ticker_b}")
            
if __name__ == "__main__":
    main()
