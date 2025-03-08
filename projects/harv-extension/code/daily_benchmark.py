import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import (
    fit_and_predict_extended, fetch_data, calculate_realized_volatility
)
from utils.models_utils import add_prime_modulo_terms, contig_prime_modulo
from utils.harvey_utils import add_harv_terms, add_harv_j_terms, add_harv_cj_terms, add_harv_tcj_terms

def plot_all_predictions(results):
    plt.figure(figsize=(14, 7))
    for strategy_name, prediction in results.items():
        plt.plot(prediction.index, prediction['Predicted'], label=f"{strategy_name} Predicted")
    # Use the Actual values from any strategy (they should all be the same)
    plt.plot(list(results.values())[0].index, list(results.values())[0]['Actual'], label="Actual", color='black', linestyle='dashed')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.title('HAR-V Model and Extended Strategies: Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def main_comparison():
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    n = 22  # Monthly window size
    warmup = 600  # Warmup period to stabilize rolling calculations

    raw_data = fetch_data(ticker, start_date, end_date)
    vol_data = calculate_realized_volatility(raw_data, n)
    
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

    plot_all_predictions(results)

if __name__ == "__main__":
    main_comparison()
