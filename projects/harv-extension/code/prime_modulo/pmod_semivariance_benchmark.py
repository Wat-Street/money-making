# For importing parent files into current files
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_utils import (
    fit_and_predict_extended, fetch_data, fetch_intraday_data,
    calculate_realized_volatility, calculate_semivariance_volatility
)
from utils.models_utils import (
    contig_prime_modulo, contig_prime_modulo_with_jumps, contig_prime_modulo_semivariance
)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch

def fit_and_predict_improved(data, features, n, warmup=30):
    """
    Improved prediction model with standardization and regularization.
    """
    predictions = []
    
    for i in range(n + warmup, len(data) - 1):
        try:
            train_data = data.iloc[:i+1].copy()
            
            # Prepare features and target
            X = train_data[features]
            y = train_data['RV_d'].shift(-1)
            X, y = X.iloc[:-1], y.iloc[:-1]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Light regularization for stability
            model = Ridge(alpha=0.1)
            model.fit(X_scaled, y)
            
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

def fit_and_predict_semivariance(data, pos_features, neg_features, n, warmup=30):
    """
    Fit separate models for positive and negative semivariance components,
    then sum the forecasts (Sum-of-the-Parts approach).
    """
    pos_predictions = []
    neg_predictions = []
    
    for i in range(n + warmup, len(data) - 1):
        try:
            train_data = data.iloc[:i+1].copy()
            
            # Prepare features and target for positive component
            X_pos = train_data[pos_features]
            y_pos = train_data['RS_pos_d'].shift(-1)
            X_pos, y_pos = X_pos.iloc[:-1], y_pos.iloc[:-1]
            
            # Prepare features and target for negative component
            X_neg = train_data[neg_features]
            y_neg = train_data['RS_neg_d'].shift(-1)
            X_neg, y_neg = X_neg.iloc[:-1], y_neg.iloc[:-1]
            
            # Scale features
            scaler_pos = StandardScaler()
            X_pos_scaled = scaler_pos.fit_transform(X_pos)
            
            scaler_neg = StandardScaler()
            X_neg_scaled = scaler_neg.fit_transform(X_neg)
            
            # Fit models with light regularization
            model_pos = Ridge(alpha=0.1)
            model_pos.fit(X_pos_scaled, y_pos)
            
            model_neg = Ridge(alpha=0.1)
            model_neg.fit(X_neg_scaled, y_neg)
            
            # Make predictions
            test_row_pos = data.iloc[[i]][pos_features].copy()
            test_row_pos_scaled = scaler_pos.transform(test_row_pos)
            pred_pos = model_pos.predict(test_row_pos_scaled).squeeze()
            
            test_row_neg = data.iloc[[i]][neg_features].copy()
            test_row_neg_scaled = scaler_neg.transform(test_row_neg)
            pred_neg = model_neg.predict(test_row_neg_scaled).squeeze()
            
            # Store predictions
            pos_predictions.append({
                'Date': data.index[i + 1],
                'Actual_Pos': data.iloc[i + 1]['RS_pos_d'],
                'Predicted_Pos': pred_pos
            })
            
            neg_predictions.append({
                'Date': data.index[i + 1],
                'Actual_Neg': data.iloc[i + 1]['RS_neg_d'],
                'Predicted_Neg': pred_neg
            })
                
        except Exception as e:
            print(f"Warning at index {i}: {str(e)}")
            continue
    
    if pos_predictions and neg_predictions:
        # Convert to DataFrames
        pos_results = pd.DataFrame(pos_predictions)
        pos_results.set_index('Date', inplace=True)
        
        neg_results = pd.DataFrame(neg_predictions)
        neg_results.set_index('Date', inplace=True)
        
        # Combine results and sum the components (SOP approach)
        combined_results = pd.concat([pos_results, neg_results], axis=1)
        combined_results['Actual_Total'] = combined_results['Actual_Pos'] + combined_results['Actual_Neg']
        combined_results['Predicted_Total'] = combined_results['Predicted_Pos'] + combined_results['Predicted_Neg']
        
        return combined_results
    else:
        return pd.DataFrame()

def compare_all_approaches():
    """
    Compares three approaches: 
    1. Regular HAR-RV
    2. Original Prime Modulo
    3. Semivariance Prime Modulo with SOP
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
    
    # Calculate standard RV and semivariance components
    vol_data = calculate_realized_volatility(raw_data, n)
    semi_vol_data = calculate_semivariance_volatility(raw_data, n)
    
    # Apply different modeling approaches
    standard_data = vol_data.copy()  # For regular HAR-RV
    original_data = contig_prime_modulo(vol_data.copy(), n)  # Original Prime Modulo
    semi_prime_data = contig_prime_modulo_semivariance(semi_vol_data.copy(), n)  # Semivariance Prime Modulo
    
    # Define features for each model
    standard_features = ['RV_d', 'RV_w', 'RV_m']  # Regular HAR-RV features
    original_features = [col for col in original_data.columns if col.startswith('RV_interval')]  # Original Prime Modulo features
    
    # Define features for SOP approach
    pos_features = [col for col in semi_prime_data.columns if col.startswith('RS_pos')]
    neg_features = [col for col in semi_prime_data.columns if col.startswith('RS_neg')]
    
    # Generate predictions
    standard_predictions = fit_and_predict_improved(standard_data, standard_features, n, warmup)
    original_predictions = fit_and_predict_improved(original_data, original_features, n, warmup)
    sop_predictions = fit_and_predict_semivariance(semi_prime_data, pos_features, neg_features, n, warmup)
    
    if standard_predictions.empty or original_predictions.empty or sop_predictions.empty:
        print("Unable to generate predictions for one or more models")
        return
    
    # Calculate error metrics
    def calculate_metrics(predictions, actual_col='Actual', pred_col='Predicted'):
        metrics = {}
        metrics['SMAPE'] = (2 * np.abs(predictions[actual_col] - predictions[pred_col]) /
                           (np.abs(predictions[actual_col]) + np.abs(predictions[pred_col]))).mean() * 100
        metrics['MAE'] = np.abs(predictions[actual_col] - predictions[pred_col]).mean()
        metrics['RMSE'] = np.sqrt(((predictions[actual_col] - predictions[pred_col]) ** 2).mean())
        return metrics
    
    standard_metrics = calculate_metrics(standard_predictions)
    original_metrics = calculate_metrics(original_predictions)
    sop_metrics = calculate_metrics(sop_predictions, 'Actual_Total', 'Predicted_Total')
    
    # Visualization
    plt.figure(figsize=(15, 15))
    
    # Plot regular HAR-RV
    plt.subplot(4, 1, 1)
    plt.plot(standard_predictions.index, standard_predictions['Actual'], 
             label='Actual', color='black', linestyle='dashed')
    plt.plot(standard_predictions.index, standard_predictions['Predicted'],
             label='Regular HAR-RV', color='purple')
    plt.title('Regular HAR-RV Model')
    plt.legend()
    plt.grid(True)
    
    # Plot original prime modulo
    plt.subplot(4, 1, 2)
    plt.plot(original_predictions.index, original_predictions['Actual'], 
             label='Actual', color='black', linestyle='dashed')
    plt.plot(original_predictions.index, original_predictions['Predicted'],
             label='Original Prime Modulo', color='blue')
    plt.title('Original Prime Modulo Model')
    plt.legend()
    plt.grid(True)
    
    # Plot SOP approach
    plt.subplot(4, 1, 3)
    plt.plot(sop_predictions.index, sop_predictions['Actual_Total'],
             label='Actual', color='black', linestyle='dashed')
    plt.plot(sop_predictions.index, sop_predictions['Predicted_Total'],
             label='SOP Prime Modulo', color='red')
    plt.title('Sum-of-the-Parts with Prime Modulo')
    plt.legend()
    plt.grid(True)
    
    # Plot performance comparison
    plt.subplot(4, 1, 4)
    
    # Align the data for comparison
    common_dates = standard_predictions.index.intersection(
        original_predictions.index.intersection(sop_predictions.index)
    )
    
    standard_error = np.abs(
        standard_predictions.loc[common_dates, 'Actual'] - 
        standard_predictions.loc[common_dates, 'Predicted']
    )
    original_error = np.abs(
        original_predictions.loc[common_dates, 'Actual'] - 
        original_predictions.loc[common_dates, 'Predicted']
    )
    sop_error = np.abs(
        sop_predictions.loc[common_dates, 'Actual_Total'] - 
        sop_predictions.loc[common_dates, 'Predicted_Total']
    )
    
    # Plot error comparison
    plt.plot(common_dates, standard_error, color='purple', label='HAR-RV Error')
    plt.plot(common_dates, original_error, color='blue', label='Original Prime Modulo Error')
    plt.plot(common_dates, sop_error, color='red', label='SOP Prime Modulo Error')
    plt.title('Error Comparison')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    print("\nMetrics Comparison:")
    print("Regular HAR-RV:")
    for metric, value in standard_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nOriginal Prime Modulo:")
    for metric, value in original_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nSOP Prime Modulo:")
    for metric, value in sop_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Calculate percentage improvement over HAR-RV
    print("\nPercentage Improvement over HAR-RV:")
    print("Original Prime Modulo vs HAR-RV:")
    for metric in standard_metrics:
        improvement = (standard_metrics[metric] - original_metrics[metric]) / standard_metrics[metric] * 100
        print(f"{metric}: {improvement:.2f}%")
    
    print("\nSOP Prime Modulo vs HAR-RV:")
    for metric in standard_metrics:
        improvement = (standard_metrics[metric] - sop_metrics[metric]) / standard_metrics[metric] * 100
        print(f"{metric}: {improvement:.2f}%")
    
    # Calculate percentage improvement of SOP over Original Prime Modulo
    print("\nSOP Prime Modulo vs Original Prime Modulo:")
    for metric in original_metrics:
        improvement = (original_metrics[metric] - sop_metrics[metric]) / original_metrics[metric] * 100
        print(f"{metric}: {improvement:.2f}%")

if __name__ == "__main__":
    compare_all_approaches()
