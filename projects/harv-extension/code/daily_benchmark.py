import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils.data_utils import (
    fit_and_predict_extended, fetch_data, calculate_realized_volatility
)
from utils.models_utils import random_sets, add_prime_modulo_terms, contig_prime_modulo, contig_prime_modulo_with_jumps
from utils.harvey_utils import add_harv_terms, add_harv_j_terms, add_harv_cj_terms, add_harv_tcj_terms
from utils.plot_utils import plot_rolling_smape, plot_regime_performance_time

def plot_daily_predictions(results, strategies):
    plt.figure(figsize=(15, 12))
    
    # Plot each model's predictions
    for i, (name, prediction) in enumerate(results.items()):
        plt.subplot(len(results) + 1, 1, i + 1)
        plt.plot(prediction.index, prediction['Actual'], 
                 label='Actual', color='black', linestyle='dashed')
        plt.plot(prediction.index, prediction['Predicted'],
                 label=name, color=f'C{i}')
        plt.legend()
        plt.grid(True)
        
        smape = (2 * np.abs(prediction['Actual'] - prediction['Predicted']) /
               (np.abs(prediction['Actual']) + np.abs(prediction['Predicted']))).mean() * 100
        plt.title(f"{name} (SMAPE: {smape:.2f}%)", fontsize=10)
    
    plt.subplot(len(results) + 1, 1, len(results) + 1)
    
    base_model = list(strategies.keys())[0]
    base_error = np.abs(results[base_model]['Actual'] - results[base_model]['Predicted'])
    
    # Plot difference between each model and the base model
    for name, prediction in results.items():
        if name == base_model:
            continue
            
        model_error = np.abs(prediction['Actual'] - prediction['Predicted'])
        error_diff = model_error - base_error
        
        # Plot line showing error difference
        plt.plot(prediction.index, error_diff, 
                 color='black', alpha=0.3, label=f'{name} vs {base_model}')
        
        # Fill areas based on which model is better
        for idx in range(len(error_diff)-1):
            current_date = prediction.index[idx]
            next_date = prediction.index[idx+1]
            current_value = error_diff.iloc[idx]
            next_value = error_diff.iloc[idx+1]
            
            if current_value >= 0:
                plt.fill_between([current_date, next_date], 
                               [current_value, next_value], 
                               [0, 0], 
                               color='red', alpha=0.3)
            else:
                plt.fill_between([current_date, next_date], 
                               [current_value, next_value], 
                               [0, 0], 
                               color='green', alpha=0.3)
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.ylabel('Error Difference\n(Model - HAR-RV)')
    
    legend_elements = [
        Patch(facecolor='red', alpha=0.3, label=f'{base_model} Better'),
        Patch(facecolor='green', alpha=0.3, label='Other Model Better')
    ]
    plt.legend(handles=legend_elements)
    plt.grid(True)
    plt.tight_layout()
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
        # "HAR-RV-TCJ": add_harv_tcj_terms,
        # "Exhaustive Search": add_exhaustive_terms,
        # "Hamming Codes": add_hamming_terms,
        "Prime Modulo Classes": add_prime_modulo_terms,
        # "Contiguous Prime Modulo": contig_prime_modulo,
        # "Contiguous CJ Prime Modulo": contig_prime_modulo_with_jumps
        "Randomized Sets": random_sets
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

    plot_daily_predictions(results, strategies)
    plot_rolling_smape(results, window_size=n)
    plot_regime_performance_time(results, window_size=n, is_intraday=False)

if __name__ == "__main__":
    main_comparison()
