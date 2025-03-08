import numpy as np

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

def contig_prime_modulo(data, n):
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
