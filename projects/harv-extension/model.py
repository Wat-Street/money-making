import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant

# Fetch data
def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    result = pd.DataFrame(index=df.index)
    result['Close'] = df['Close']
    result['Log_Return'] = np.log(result['Close'] / result['Close'].shift(1))
    result['Squared_Return'] = result['Log_Return'] ** 2
    return result.dropna()

# Calculate realized volatility
def calculate_realized_volatility(df, n):
    result = pd.DataFrame(index=df.index)
    result['RV_d'] = df['Squared_Return']
    result['RV_w'] = df['Squared_Return'].rolling(window=5).mean()
    result['RV_m'] = df['Squared_Return'].rolling(window=n).mean()
    return result.dropna()

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

# Strategy 3: Prime Modulo Classes
def add_prime_modulo_terms(data, n):
    data = data.reset_index()
    data['Index'] = range(len(data))
    primes = []
    candidate = 2
    while np.prod(primes, dtype=np.int64) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    for prime in primes:
        col_name = f"RV_mod_{prime}"
        data[col_name] = ((data['Index'] % prime == 0).astype(int)) * data['RV_d']
    return data.set_index('Date')

# Standard HAR-RV Model
def add_harv_terms(data, n):
    return data

# Prediction Model
def fit_and_predict_extended(data, features, n, warmup=30):
    predictions = []
    for i in range(n + warmup, len(data) - 1):
        try:
            train_data = data.iloc[:i+1].copy()
            X = train_data[features]
            y = train_data['RV_d'].shift(-1)
            X, y = X.iloc[:-1], y.iloc[:-1]
            X = add_constant(X)
            model = OLS(y, X).fit()
            test_row = data.iloc[[i]][features].copy()
            test_row = add_constant(test_row, has_constant='add')
            test_row = test_row.reindex(columns=X.columns, fill_value=0)
            pred = model.predict(test_row).squeeze()
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

    raw_data = fetch_data(ticker, start_date, end_date)
    vol_data = calculate_realized_volatility(raw_data, n)

    strategies = {
        "Standard HAR-RV": add_harv_terms,
        "Exhaustive Search": add_exhaustive_terms,
        "Hamming Codes": add_hamming_terms,
        "Prime Modulo Classes": add_prime_modulo_terms,
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
