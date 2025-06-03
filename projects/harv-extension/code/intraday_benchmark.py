import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from utils.data_utils import (
    fit_and_predict_extended, fetch_intraday_data, calculate_intraday_realized_volatility
)
from utils.models_utils import random_sets, add_prime_modulo_terms, contig_prime_modulo
from utils.harvey_utils import add_harv_terms, add_harv_j_terms, add_harv_cj_terms, add_harv_tcj_terms
from utils.plot_utils import plot_rolling_smape, plot_regime_performance_time

def plot_intraday_predictions(results, days_to_show=1):
    first_result = list(results.values())[0]
    
    points_per_day = 288  # Assuming 5-minute data (12 points/hour * 24 hours)
    points_to_show = min(points_per_day * days_to_show, len(first_result))
    start_idx = max(0, len(first_result) - points_to_show)
    
    n_plots = len(results) + 1
    _, axes = plt.subplots(n_plots, 1, figsize=(15, 4*n_plots), sharex=True)
    
    if n_plots == 2:
        axes = [axes[0], axes[1]]
    
    # Plot each model on its own subplot
    for i, (strategy_name, prediction) in enumerate(results.items()):
        ax = axes[i]
        ax.plot(prediction.index[start_idx:], prediction['Predicted'].iloc[start_idx:],
                label=f"Predicted", color=f'C{i}', linewidth=1.5)
        ax.plot(first_result.index[start_idx:], first_result['Actual'].iloc[start_idx:],
                label="Actual", color='black', linestyle='dashed', linewidth=1.5)
        
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_minor_locator(mdates.HourLocator())
        
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Volatility')
        
        data_subset = prediction.iloc[start_idx:]
        smape = (2 * np.abs(data_subset['Actual'] - data_subset['Predicted']) /
                (np.abs(data_subset['Actual']) + np.abs(data_subset['Predicted']))).mean() * 100
        ax.set_title(f"{strategy_name} (SMAPE: {smape:.2f}%)", fontsize=10)
        
        # Add legend
        ax.legend(loc='upper right')
    
    base_model = list(results.keys())[0]
    base_error = np.abs(results[base_model]['Actual'].iloc[start_idx:] - 
                         results[base_model]['Predicted'].iloc[start_idx:])
    comp_ax = axes[-1]
    
    for i, (name, prediction) in enumerate(results.items()):
        if name == base_model:
            continue
            
        # Calculate error difference
        model_error = np.abs(prediction['Actual'].iloc[start_idx:] - 
                              prediction['Predicted'].iloc[start_idx:])
        error_diff = model_error - base_error
        
        # Plot the error difference line
        comp_ax.plot(prediction.index[start_idx:], error_diff, 
                    color='black', alpha=0.5, label=f'{name} vs {base_model}')
        
        # Fill areas based on which model performs better
        for idx in range(len(error_diff)-1):
            current_date = prediction.index[start_idx+idx]
            next_date = prediction.index[start_idx+idx+1]
            current_value = error_diff.iloc[idx]
            next_value = error_diff.iloc[idx+1]
            
            if current_value >= 0:
                comp_ax.fill_between([current_date, next_date], 
                                   [current_value, next_value], 
                                   [0, 0], 
                                   color='red', alpha=0.3)
            else:
                comp_ax.fill_between([current_date, next_date], 
                                   [current_value, next_value], 
                                   [0, 0], 
                                   color='green', alpha=0.3)
    
    # Format comparison subplot
    comp_ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    comp_ax.set_ylabel('Error Difference\n(Model - Base)')
    
    # Add legend for comparison plot
    legend_elements = [
        Patch(facecolor='red', alpha=0.3, label=f'{base_model} Better'),
        Patch(facecolor='green', alpha=0.3, label='Other Model Better')
    ]
    comp_ax.legend(handles=legend_elements)
    comp_ax.grid(True, alpha=0.3)
    
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        comp_ax.set_xticklabels([])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

def main_comparison():
    n = 22  # Monthly window size
    warmup = 600  # Warmup period to stabilize rolling calculations

    raw_data = fetch_intraday_data()
    vol_data = calculate_intraday_realized_volatility(raw_data)

    strategies = {
        "Standard HAR-RV": add_harv_terms,
        # "HAR-RV-J": add_harv_j_terms,
        # "HAR-RV-CJ": add_harv_cj_terms,
        # "HAR-RV-TCJ": add_harv_tcj_terms,
        # "Exhaustive Search": add_exhaustive_terms,
        # "Hamming Codes": add_hamming_terms,
        # "Prime Modulo Classes": add_prime_modulo_terms,
        "Randomized Sets": random_sets,
        # "Contiguous Prime Modulo": contig_prime_modulo
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
    
    plot_intraday_predictions(results, days_to_show=1)
    plot_rolling_smape(results, window_size=288)
    plot_regime_performance_time(results, window_size=288, is_intraday=True)

if __name__ == "__main__":
    main_comparison()
