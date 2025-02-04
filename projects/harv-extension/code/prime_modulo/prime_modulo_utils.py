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
