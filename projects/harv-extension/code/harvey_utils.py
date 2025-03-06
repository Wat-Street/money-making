import numpy as np
from scipy import stats
import pandas as pd

# Standard HAR-RV Model
def add_harv_terms(data, n):
    return data

# HAR-RVJ Model
def add_harv_j_terms(data, n):
    # Bipower variation (BV)
    const = np.sqrt(2/np.pi)
    abs_returns = np.sqrt(data['RV_d']).shift(1).abs()
    data['BV_d'] = (const * abs_returns * np.sqrt(data['RV_d'])).fillna(0)
    
    # Jump component (max(RV - BV, 0)
    data['J_d'] = np.maximum(0, data['RV_d'] - data['BV_d'])
    
    data['BV_w'] = data['BV_d'].rolling(window=5).mean()
    data['BV_m'] = data['BV_d'].rolling(window=n).mean()
    
    data['J_w'] = data['J_d'].rolling(window=5).mean()
    data['J_m'] = data['J_d'].rolling(window=n).mean()
    
    data['RV_BV_d'] = data['BV_d']
    data['RV_BV_w'] = data['BV_w']
    data['RV_BV_m'] = data['BV_m']
    data['RV_J_d'] = data['J_d']
    data['RV_J_w'] = data['J_w']
    data['RV_J_m'] = data['J_m']
    
    return data.dropna()

# HAR-RVCJ Model
def add_harv_cj_terms(data, n, alpha=0.999):
    # Bipower variation (BV)
    const = np.sqrt(2/np.pi)
    abs_returns = np.sqrt(data['RV_d']).shift(1).abs()
    data['BV_d'] = (const * abs_returns * np.sqrt(data['RV_d'])).fillna(0)
    
    # Integrated quarticity (for jump test statistic)
    data['IQ_d'] = (data['RV_d'] ** 2) * np.pi/2
    
    # Test statistic for jump detection
    data['z_stat'] = np.sqrt(252) * (data['RV_d'] - data['BV_d']) / np.sqrt(data['IQ_d'])
    
    # Identify jump days using statistical threshold
    critical_value = stats.norm.ppf(alpha)
    data['jump_day'] = data['z_stat'] > critical_value
    
    # Decompose -> continuous and jump components
    # On jump days, C = BV and J = RV-BV
    # On non-jump days, C = RV and J = 0
    data['C_d'] = np.where(data['jump_day'], 
                             data['BV_d'], 
                             data['RV_d'])
    data['J_d'] = np.where(data['jump_day'], 
                             data['RV_d'] - data['BV_d'], 
                             0)
    
    data['C_w'] = data['C_d'].rolling(window=5).mean()
    data['C_m'] = data['C_d'].rolling(window=n).mean()
    
    data['J_w'] = data['J_d'].rolling(window=5).mean()
    data['J_m'] = data['J_d'].rolling(window=n).mean()
    
    data['RV_C_d'] = data['C_d']
    data['RV_C_w'] = data['C_w']
    data['RV_C_m'] = data['C_m']
    data['RV_J_d'] = data['J_d']
    data['RV_J_w'] = data['J_w']
    data['RV_J_m'] = data['J_m']
    
    return data.dropna()

def add_harv_tcj_terms(data, n, alpha=0.999):
    # bipower variation
    const = np.sqrt(2/np.pi)
    abs_returns = np.sqrt(data['RV_d']).shift(1).abs()
    data['BV_d'] = (const * abs_returns * np.sqrt(data['RV_d'])).fillna(0)
    
    # calculate tripower quarticity
    u_43 = 2**(2/3) * (np.pi**(1/3)) / (4**(2/3) * (np.pi-2)**(1/3))
    abs_returns_power = abs_returns**(4/3)
    
    shifts = [0, 1, 2]
    product = pd.Series(1, index=abs_returns.index)
    
    for shift in shifts:
        if shift > 0:
            product *= abs_returns_power.shift(shift)
        else:
            product *= abs_returns_power
    
    data['TQ_d'] = u_43 * (product**(3/4))
    
    # threshold Z-statistic (CT_Z)
    delta = 1/252
    data['z_stat'] = (data['RV_d'] - data['BV_d']) / (
        np.sqrt(delta * ((np.pi**2)/4 + np.pi - 5) * np.maximum(1, data['TQ_d']/(data['BV_d']**2)))
    )
    
    # threshold correction
    theta = 0.3  # threshold parameter from paper
    data['CT_Z'] = data['z_stat'] * (data['RV_d'] > theta * data['BV_d'])
    
    # Identify jump days
    critical_value = stats.norm.ppf(alpha)
    data['jump_day'] = data['CT_Z'] > critical_value
    
    # Separate continuous and jump components
    data['TCJ_d'] = np.where(data['jump_day'],
                             data['RV_d'] - data['BV_d'],
                             0)
    data['TC_d'] = np.where(data['jump_day'],
                            data['BV_d'],
                            data['RV_d'])
    
    data['TC_w'] = data['TC_d'].rolling(window=5).mean()
    data['TC_m'] = data['TC_d'].rolling(window=n).mean()
    data['TCJ_w'] = data['TCJ_d'].rolling(window=5).mean()
    data['TCJ_m'] = data['TCJ_d'].rolling(window=n).mean()
    
    data['RV_TC_d'] = data['TC_d']
    data['RV_TC_w'] = data['TC_w']
    data['RV_TC_m'] = data['TC_m']
    data['RV_TCJ_d'] = data['TCJ_d']
    data['RV_TCJ_w'] = data['TCJ_w']
    data['RV_TCJ_m'] = data['TCJ_m']
    
    return data.dropna()
