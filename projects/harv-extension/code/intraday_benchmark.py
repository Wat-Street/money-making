import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils.data_utils import (
    fit_and_predict_extended, fetch_intraday_data, calculate_intraday_realized_volatility
)
from utils.models import add_prime_modulo_terms, contig_prime_modulo
from utils.harvey_utils import add_harv_terms, add_harv_j_terms, add_harv_cj_terms, add_harv_tcj_terms

# Strategy 1: Exhaustive Search
def add_exhaustive_terms(data, n):
    for j in range(1, n + 1):
        col_name = f"RV_{j}"
        data[col_name] = data['RV_d'].rolling(window=j).mean()
    return data.replace([np.inf, -np.inf], np.nan).dropna()

# Strategy 2: Hamming Codes
def add_hamming_terms(data, n):
    data = data.reset_index()
    data['Index'] = range(len(data))
    num_terms = int(np.ceil(np.log2(n)))
    for j in range(num_terms):
        col_name = f"RV_bin_{j}"
        data[col_name] = ((data['Index'] & (1 << j)) != 0).astype(int) * data['RV_d']
    return data.set_index('Date')

def plot_intraday_predictions(results, days_to_show=1):
    first_result = list(results.values())[0]
    
    points_per_day = 288  # Assuming 5-minute data (12 points/hour * 24 hours)
    points_to_show = min(points_per_day * days_to_show, len(first_result))
    start_idx = max(0, len(first_result) - points_to_show)
    
    plt.figure(figsize=(15, 8))
    for strategy_name, prediction in results.items():
        plt.plot(prediction.index[start_idx:], prediction['Predicted'].iloc[start_idx:],
                label=f"{strategy_name} Predicted", linewidth=1)
    
    plt.plot(first_result.index[start_idx:], first_result['Actual'].iloc[start_idx:],
            label="Actual", color='black', linestyle='dashed', linewidth=1.5)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))  # Every 4 hours
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_minor_locator(mdates.HourLocator())
    
    plt.grid(True, alpha=0.3)
    
    plt.xlabel('Time')
    plt.ylabel('Intraday Volatility')
    plt.title(f'{days_to_show}-Day Intraday Volatility Prediction')
    
    # Get SMAPE metrics
    performance_text = "SMAPE:\n"
    for strategy_name, prediction in results.items():
        data_subset = prediction.iloc[start_idx:]
        smape = (2 * np.abs(data_subset['Actual'] - data_subset['Predicted']) /
                (np.abs(data_subset['Actual']) + np.abs(data_subset['Predicted']))).mean() * 100
        performance_text += f"{strategy_name}: {smape:.2f}%\n"
    
    plt.figtext(0.02, 0.02, performance_text, 
                bbox=dict(facecolor='white', alpha=0.9, boxstyle="round,pad=0.5"))
    
    plt.legend(loc='upper right')
    plt.gcf().autofmt_xdate(rotation=30)
    plt.tight_layout()
    plt.show()

def main_comparison():
    n = 22  # Monthly window size
    warmup = 600  # Warmup period to stabilize rolling calculations

    raw_data = fetch_intraday_data()
    vol_data = calculate_intraday_realized_volatility(raw_data)

    strategies = {
        # "Standard HAR-RV": add_harv_terms,
        # "HAR-RV-J": add_harv_j_terms,
        # "HAR-RV-CJ": add_harv_cj_terms,
        "HAR-RV-TCJ": add_harv_tcj_terms,
        # "Exhaustive Search": add_exhaustive_terms,
        # "Hamming Codes": add_hamming_terms,
        # "Prime Modulo Classes": add_prime_modulo_terms,
        "Contiguous Prime Modulo": contig_prime_modulo
    }

    results = {}
    for name, strategy in strategies.items():
        print(f"\nRunning strategy: {name}")
        extended_data = strategy(vol_data.copy(), n)
        features = [col for col in extended_data.columns if col.startswith('RV')]
        predictions = fit_and_predict_extended(extended_data, features, n, warmup)
        if not predictions.empty:
            smape = (2 * np.abs(predictions['Actual'] - predictions['Predicted']) /
                    (np.abs(predictions['Actual']) + np.abs(predictions['Predicted']))).mean() * 100
            print(f"{name} SMAPE: {smape:.2f}%")
            results[name] = predictions

    plot_intraday_predictions(results)

if __name__ == "__main__":
    main_comparison()
