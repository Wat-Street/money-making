import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant

# Calculate realized volatility
def calculate_realized_volatility(df, n):
    result = pd.DataFrame(index=df.index)
    result['RV_d'] = df['Squared_Return']
    result['RV_w'] = df['Squared_Return'].rolling(window=5).mean()
    result['RV_m'] = df['Squared_Return'].rolling(window=n).mean()
<<<<<<< HEAD
    result['Volume'] = df['Volume']
=======
>>>>>>> cc5eefc (harv-dev-prime-modulo: adding prime modulo improvements)
    return result.dropna()

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
<<<<<<< HEAD

def contig_prime_modulo(data, n):
    original_index = data.index
    data = data.reset_index()
    index_col = 'Date' if 'Date' in data.columns else data.columns[0]
    data['Index'] = range(len(data))

    primes = []
    candidate = 2
    while np.prod(primes, dtype=np.int64) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    
    print(f'utilizing {len(primes)} primes: {primes}')
    
    for prime in primes:
        prime_indices = data.index[data['Index'] % prime == 0].tolist()
        col_name = f"RV_interval_{prime}"
        data[col_name] = 0.0
        
        # Calculate volatility for each interval
        for i in range(len(prime_indices) - 1):
            start_idx = prime_indices[i]
            end_idx = prime_indices[i+1]
            
            # Several options for calculating interval volatility:
            
            # Option 1: Average RV over the interval
            interval_rv = data.loc[start_idx:end_idx, 'RV_d'].mean()
            
            # Option 2: Cumulative RV over the interval
            # interval_rv = data.loc[start_idx:end_idx, 'RV_d'].sum()
            
            # Option 3: Standard deviation of returns over the interval
            # This would require having the original returns, not just RV
            
            data.loc[start_idx:end_idx, col_name] = interval_rv
    
    # Set the index back to the original time-based index
    return data.set_index(index_col)
=======
>>>>>>> cc5eefc (harv-dev-prime-modulo: adding prime modulo improvements)
