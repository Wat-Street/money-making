from datetime import datetime
import time
from analysis.data_loader import load_pair_data
from analysis.visualizer import plot_lagged_correlation
from analysis.pairs import research_pairs

def main():
    start_date = "2020-01-01" # approx 5 years but we can change this later
    end_date = datetime.today.strftime("%Y-%m-%d")

    print(f"Starting correlation analysis from {start_date} to {end_date}")
    print(f"Processing {len(research_pairs)} pairs...")
    success = 0

    for i, (ticker_a, ticker_b) in enumerate(research_pairs, 1):
        print(f"\n{'='*50}")
        print(f"Processing pair {i}/{len(research_pairs)}: {ticker_a} & {ticker_b}")
        print(f"{'='*50}")

        try:
            df = load_pair_data(ticker_a, ticker_b, start_date, end_date, cache=True)
            if df.empty:
                print(f"❌ No data for {ticker_a} and {ticker_b}")
                continue

            print(f"✅ Successfully loaded data for {ticker_a}-{ticker_b} ")
            plot_lagged_correlation(ticker_a, ticker_b, df, max_lag=10, save=True)
            success += 1
        
        except Exception as e:
            print(f"❌ Error processing {ticker_a} and {ticker_b}")
            continue
        
        # Stagger API calls for stability
        if i < len(research_pairs):
            print("Waiting 2 seconds before next pair...")
            time.sleep(2)

    print(f"\n{'='*50}")
    print(f"Successfully processed: {success}/{len(research_pairs)} pairs")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()