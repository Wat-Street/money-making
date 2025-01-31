# For importing parent files into current files
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
from data_and_prediction_utils import fit_and_predict_extended, fetch_data
from prime_modulo_utils import calculate_realized_volatility, add_prime_modulo_terms

# Strategy 3: # Improved prime modulo implementation incorporating our enhancements 
def add_improved_prime_modulo_terms(data, n):
    data = data.reset_index()
    data['Index'] = range(len(data))
    
    # Refinement 1: More Flexible Prime Selection
    # Instead of fixed market primes, we'll use a hybrid approach that combines
    # market-relevant primes with discovered ones
    base_market_primes = [2, 5, 23]  # Reduced set of core market cycle primes
    
    # Find additional primes up to n that might capture other cycles
    additional_primes = []
    candidate = 2
    while candidate <= n:
        if all(candidate % p != 0 for p in base_market_primes + additional_primes):
            additional_primes.append(candidate)
        candidate += 1
    
    # Combine both sets but limit total number to control complexity
    max_primes = 5  # Limit total number of primes to reduce parameter space
    selected_primes = base_market_primes + additional_primes[:max_primes - len(base_market_primes)]
    
    # Refinement 2: Simplified Regime Detection
    # Use only two regimes instead of three to reduce complexity
    vol_std = data['RV_d'].rolling(window=22).std()
    vol_mean = data['RV_d'].rolling(window=22).mean()
    
    # More conservative regime threshold
    data['regime'] = 0  # normal volatility
    data.loc[data['RV_d'] > (vol_mean + 1.5 * vol_std), 'regime'] = 1  # high volatility only
    
    # Refinement 3: Gentler Decay
    alpha = 0.98  # Increased from 0.95 to preserve more historical information
    
    # Create weighted modulo terms with reduced complexity
    for prime in selected_primes:
        for remainder in range(prime):
            base_mask = (data['Index'] % prime == remainder)
            
            # Calculate base weights first
            weights = np.zeros(len(data))
            relevant_indices = data[base_mask].index
            
            for current_idx in relevant_indices:
                current_time = data.loc[current_idx, 'Index']
                time_diff = abs(current_time - data.loc[current_idx, 'Index'])
                weights[current_idx] = alpha ** time_diff
            
            # Create base term without regime dependence
            base_col = f"RV_mod_{prime}_{remainder}"
            data[base_col] = data['RV_d'] * weights
            
            # Add regime-specific terms only for high volatility
            # This reduces parameters while still capturing extreme events
            if prime in base_market_primes:  # Only apply regime split to market-relevant primes
                high_vol_col = f"{base_col}_highvol"
                data[high_vol_col] = data[base_col] * (data['regime'] == 1).astype(float)
    
    return data.set_index('Date')

# Modified prediction function to handle regime-dependent parameters
def fit_and_predict_improved(data, features, n, warmup=30):
    predictions = []
    
    for i in range(n + warmup, len(data) - 1):
        try:
            train_data = data.iloc[:i+1].copy()
            
            # Use all features for prediction but with regularization
            X = train_data[features]
            y = train_data['RV_d'].shift(-1)
            X, y = X.iloc[:-1], y.iloc[:-1]
            
            # Add L2 regularization through Ridge Regression
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=0.1)  # Light regularization
            
            # Standardize features for better regression
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model.fit(X_scaled, y)
            
            # Prepare test data
            test_row = data.iloc[[i]][features].copy()
            test_row_scaled = scaler.transform(test_row)
            
            pred = model.predict(test_row_scaled).squeeze()
            
            predictions.append({
                'Date': data.index[i + 1],
                'Actual': data.iloc[i + 1]['RV_d'],
                'Predicted': pred,
                'Regime': data.iloc[i]['regime']
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
    # Set up test parameters
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    n = 22  # Monthly window size
    warmup = 600
    
    # Fetch and prepare data
    raw_data = fetch_data(ticker, start_date, end_date)
    vol_data = calculate_realized_volatility(raw_data, n)
    
    # Run both versions
    original_data = add_prime_modulo_terms(vol_data.copy(), n)
    improved_data = add_improved_prime_modulo_terms(vol_data.copy(), n)
    
    # Get predictions
    original_features = [col for col in original_data.columns if col.startswith('RV')]
    improved_features = [col for col in improved_data.columns if col.startswith('RV')]
    
    original_predictions = fit_and_predict_extended(original_data, original_features, n, warmup)
    improved_predictions = fit_and_predict_improved(improved_data, improved_features, n, warmup)
    
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
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot original predictions
    plt.subplot(2, 1, 1)
    plt.plot(original_predictions.index, original_predictions['Actual'], 
             label='Actual', color='black', linestyle='dashed')
    plt.plot(original_predictions.index, original_predictions['Predicted'],
             label='Original Prime Modulo', color='blue')
    plt.title('Original Prime Modulo Predictions')
    plt.legend()
    plt.grid(True)
    
    # Plot improved predictions with safer regime background handling
    plt.subplot(2, 1, 2)
    plt.plot(improved_predictions.index, improved_predictions['Actual'],
             label='Actual', color='black', linestyle='dashed')
    plt.plot(improved_predictions.index, improved_predictions['Predicted'],
             label='Improved Prime Modulo', color='red')
    
    # Safer regime background plotting
    if 'Regime' in improved_predictions.columns:
        # We now only have 2 regimes (0 and 1)
        for regime in [0, 1]:
            mask = improved_predictions['Regime'] == regime
            if any(mask):  # Only plot if we have data for this regime
                regime_periods = improved_predictions[mask].index
                if len(regime_periods) > 0:
                    # Use different colors for different regimes
                    color = 'lightblue' if regime == 0 else 'salmon'
                    alpha = 0.2
                    
                    # Plot each continuous period of the regime
                    regime_changes = np.where(np.diff(mask.astype(int)))[0]
                    start_idx = 0
                    
                    for end_idx in regime_changes:
                        if mask.iloc[start_idx]:
                            plt.axvspan(regime_periods[start_idx],
                                      regime_periods[end_idx],
                                      alpha=alpha, color=color)
                        start_idx = end_idx + 1
                    
                    # Don't forget the last period
                    if mask.iloc[start_idx]:
                        plt.axvspan(regime_periods[start_idx],
                                  regime_periods[-1],
                                  alpha=alpha, color=color)
    
    plt.title('Improved Prime Modulo Predictions')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics comparison
    print("\nMetrics Comparison:")
    print("Original Prime Modulo:")
    for metric, value in original_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\nImproved Prime Modulo:")
    for metric, value in improved_metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    compare_prime_modulo_versions()