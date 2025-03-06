# For importing parent files into current files
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
from data_and_prediction_utils import fit_and_predict_extended, fetch_data, fetch_intraday_data
from prime_modulo_utils import calculate_realized_volatility, add_prime_modulo_terms, contig_prime_modulo

def add_volume_weighted_prime_modulo_terms(data, n):
    data = data.reset_index()
    data['Index'] = range(len(data))
    
    data['normalized_volume'] = (
        data['Volume'] / data['Volume'].rolling(window=n, min_periods=1).mean()
    )
    
    # Core market cycle primes
    base_market_primes = [2, 5, 23]  # daily, weekly, monthly
    
    # Find additional primes up to n, prioritizing those close to known market cycles
    additional_primes = []
    candidate = 2
    while candidate <= n:
        if all(candidate % p != 0 for p in base_market_primes + additional_primes):
            additional_primes.append(candidate)
        candidate += 1
    
    # Combine both sets but prioritize market-aligned primes
    all_primes = base_market_primes.copy()
    for p in additional_primes:
        if len(all_primes) < 5 and p not in all_primes:
            all_primes.append(p)

    print(f'utilizing {len(all_primes)} primes: {all_primes}')
    for prime in all_primes:
        col_name = f"RV_mod_{prime}"
        data[col_name] = (
            (data['Index'] % prime == 0).astype(int) * 
            data['RV_d'] * 
            data['normalized_volume']
        )
    
    return data.set_index('Date')

def add_volume_weighted_adaptive_prime_modulo_terms(data, n):
    data = data.reset_index()
    data['Index'] = range(len(data))
    
    # Calculate normalized volume for direct weighting
    data['normalized_volume'] = (
        data['Volume'] / data['Volume'].rolling(window=n, min_periods=1).mean()
    )
    
    # Calculate market stress indicators
    data['vol_level'] = data['RV_d'] / data['RV_d'].rolling(window=n).mean()
    data['vol_of_vol'] = data['RV_d'].rolling(window=n).std() / data['RV_d'].rolling(window=n).std()
    data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(window=n).mean()
    
    # Combine into market stress indicator
    data['market_stress'] = (
        (data['vol_level'] + data['vol_of_vol'] + data['volume_ratio']) / 3
    ).clip(0, 1)
    
    calm_market_primes = [7, 23]
    stress_market_primes = [2, 3, 5]
    
    all_possible_primes = set(calm_market_primes + stress_market_primes)
    for prime in all_possible_primes:
        data[f"RV_mod_{prime}"] = 0
    
    for idx in data.index:
        stress_level = data.loc[idx, 'market_stress']
        selected_primes = []
        
        for i in range(min(len(calm_market_primes), len(stress_market_primes))):
            if stress_level > 0.7:           # High stress regime
                selected_primes.append(stress_market_primes[i])
            elif stress_level < 0.3:         # Low stress regime
                selected_primes.append(calm_market_primes[i])
            else:                            # Medium stress - mix of both
                selected_primes.append(
                    stress_market_primes[i] if i < 2 else calm_market_primes[i]
                )
        
        current_volume_weight = data.loc[idx, 'normalized_volume']
        
        for prime in selected_primes:
            col_name = f"RV_mod_{prime}"
            data.loc[idx, col_name] = (
                (data.loc[idx, 'Index'] % prime == 0).astype(int) *
                data.loc[idx, 'RV_d'] *
                current_volume_weight
            )
    
    return data.set_index('Date')

def sliding_window_prime_modulo(data, n):
    # Create windows of n days and flatten for volatility calculation
    windows = []
    for i in range(len(data) - n + 1):
        window = data.iloc[i:i+n]
        windows.append(window)
    
    base_market_primes = [2, 5, 23]  # daily, weekly, monthly
    
    for prime in base_market_primes:
        vol_values = []
        for i in range(len(data)):
            if i % prime == 0 and i < len(windows):
                window_data = windows[i]
                vol = np.sqrt(np.sum(window_data['Squared_Return']))
                vol_values.append(vol)
            else:
                vol_values.append(np.nan)
                
        data[f'RV_mod_{prime}'] = vol_values
        data[f'RV_mod_{prime}'].fillna(method='ffill', inplace=True)
        
    return data

def fit_and_predict_improved(data, features, n, warmup=30):
    """
    Enhanced prediction function using Ridge regression and feature standardization.
    Removes regime-dependent parameters for more stable predictions.
    """
    predictions = []
    
    for i in range(n + warmup, len(data) - 1):
        try:
            train_data = data.iloc[:i+1].copy()
            
            # Prepare features and target
            X = train_data[features]
            y = train_data['RV_d'].shift(-1)
            X, y = X.iloc[:-1], y.iloc[:-1]
            
            # Standardize features and apply Ridge regression
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Light regularization for stability
            model = Ridge(alpha=0.1)
            model.fit(X_scaled, y)
            
            # Make prediction
            test_row = data.iloc[[i]][features].copy()
            test_row_scaled = scaler.transform(test_row)
            pred = model.predict(test_row_scaled).squeeze()
            
            predictions.append({
                'Date': data.index[i + 1],
                'Actual': data.iloc[i + 1]['RV_d'],
                'Predicted': pred
            })
                
        except Exception as e:
            print(f"Warning at index {i}: {str(e)}")
            continue
    
    if predictions:
        results = pd.DataFrame(predictions)
        results.set_index('Date', inplace=True)
        return results
    else:
        return pd.DataFrame()

def compare_prime_modulo_versions():
    """
    Compares original and improved prime modulo approaches with simplified visualization.
    """
    # Set up test parameters
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    n = 22  # Monthly window size
    warmup = 600
    
    # Fetch and prepare data
    raw_data = fetch_data(ticker, start_date, end_date)
    if raw_data.empty:
        print(f"No data found for {ticker} between {start_date} and {end_date}")
        return
    
    raw_intraday_data = fetch_intraday_data()
    if raw_intraday_data.empty:
        print(f"No intraday data found")
        return
    
    vol_data = calculate_realized_volatility(raw_data, n)
    original_data = add_prime_modulo_terms(vol_data.copy(), n)
    improved_data = contig_prime_modulo(vol_data.copy(), n)
    
    original_features = [col for col in original_data.columns if col.startswith('RV')]
    improved_features = [col for col in improved_data.columns if col.startswith('RV')]
    
    original_predictions = fit_and_predict_extended(original_data, original_features, n, warmup)
    improved_predictions = fit_and_predict_improved(improved_data, improved_features, n, warmup)
    
    if original_predictions.empty or improved_predictions.empty:
        print("Unable to generate predictions for one or both models")
        return
    
    # Calculate error metrics
    def calculate_metrics(predictions):
        metrics = {}
        metrics['SMAPE'] = (2 * np.abs(predictions['Actual'] - predictions['Predicted']) /
                           (np.abs(predictions['Actual']) + np.abs(predictions['Predicted']))).mean() * 100
        metrics['MAE'] = np.abs(predictions['Actual'] - predictions['Predicted']).mean()
        metrics['RMSE'] = np.sqrt(((predictions['Actual'] - predictions['Predicted']) ** 2).mean())
        return metrics
    
    original_metrics = calculate_metrics(original_predictions)
    improved_metrics = calculate_metrics(improved_predictions)
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 1, 1)
    plt.plot(original_predictions.index, original_predictions['Actual'], 
             label='Actual', color='black', linestyle='dashed')
    plt.plot(original_predictions.index, original_predictions['Predicted'],
             label='Original Prime Modulo', color='blue')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(improved_predictions.index, improved_predictions['Actual'],
             label='Actual', color='black', linestyle='dashed')
    plt.plot(improved_predictions.index, improved_predictions['Predicted'],
             label='Improved Prime Modulo', color='red')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    
    original_error = np.abs(original_predictions['Actual'] - original_predictions['Predicted'])
    improved_error = np.abs(improved_predictions['Actual'] - improved_predictions['Predicted'])
    
    relative_improvement = improved_error - original_error
    
    plt.plot(original_predictions.index, relative_improvement, 
             color='black', alpha=0.5, label='Performance Difference')
    
    for idx in range(len(relative_improvement)-1):
        current_date = original_predictions.index[idx]
        next_date = original_predictions.index[idx+1]
        current_value = relative_improvement.iloc[idx]
        next_value = relative_improvement.iloc[idx+1]
        
        # Red areas show where original model performs better
        if current_value >= 0:
            plt.fill_between([current_date, next_date], 
                           [current_value, next_value], 
                           [0, 0], 
                           color='red', alpha=0.3)
        # Green areas show where improved model performs better
        else:
            plt.fill_between([current_date, next_date], 
                           [current_value, next_value], 
                           [0, 0], 
                           color='green', alpha=0.3)
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.ylabel('Error Difference (Improved - Original)')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.3, label='Original Version Better'),
        Patch(facecolor='green', alpha=0.3, label='Improved Version Better')
    ]
    plt.legend(handles=legend_elements)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("\nMetrics Comparison:")
    print("Original Prime Modulo:")
    for metric, value in original_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\nImproved Prime Modulo:")
    for metric, value in improved_metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    compare_prime_modulo_versions()