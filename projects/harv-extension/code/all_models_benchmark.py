import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
<<<<<<< HEAD
from data_and_prediction_utils import (
    fit_and_predict_extended, fetch_data, fetch_intraday_data, calculate_realized_volatility, calculate_intraday_realized_volatility
)
from prime_modulo.prime_modulo_utils import add_prime_modulo_terms, contig_prime_modulo
from harvey_utils import add_harv_terms, add_harv_j_terms, add_harv_cj_terms, add_harv_tcj_terms
=======
from data_and_prediction_utils import fit_and_predict_extended
from data_and_prediction_utils import fetch_data 
from prime_modulo.prime_modulo_utils import calculate_realized_volatility, add_prime_modulo_terms
>>>>>>> cc5eefc (harv-dev-prime-modulo: adding prime modulo improvements)

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

# Standard HAR-RV Model
def add_harv_terms(data, n):
    return data

# Plot Predictions for All Strategies
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

# Main Comparison Function
def main_comparison():
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    n = 22  # Monthly window size
    warmup = 600  # Warmup period to stabilize rolling calculations

<<<<<<< HEAD
    raw_data = fetch_intraday_data()
    vol_data = calculate_intraday_realized_volatility(raw_data)

    strategies = {
        "Standard HAR-RV": add_harv_terms,
        # "Exhaustive Search": add_exhaustive_terms,
        # "Hamming Codes": add_hamming_terms,
        # "Prime Modulo Classes": add_prime_modulo_terms,
        "Contiguous Prime Modulo": contig_prime_modulo
=======
    raw_data = fetch_data(ticker, start_date, end_date)
    vol_data = calculate_realized_volatility(raw_data, n)

    strategies = {
        "Standard HAR-RV": add_harv_terms,
        "Exhaustive Search": add_exhaustive_terms,
        "Hamming Codes": add_hamming_terms,
        "Prime Modulo Classes": add_prime_modulo_terms,
>>>>>>> cc5eefc (harv-dev-prime-modulo: adding prime modulo improvements)
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
