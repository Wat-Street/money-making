import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_rolling_smape(results, window_size=24):
    plt.figure(figsize=(15, 6))
    
    for strategy_name, prediction in results.items():
        # Calculate absolute percentage error for each point
        ape = (2 * np.abs(prediction['Actual'] - prediction['Predicted']) /
              (np.abs(prediction['Actual']) + np.abs(prediction['Predicted'])))
        
        # Calculate rolling mean with specified window
        rolling_smape = ape.rolling(window=window_size).mean() * 100
        
        # Plot the rolling SMAPE
        plt.plot(prediction.index, rolling_smape, label=f"{strategy_name}")
    
    # Add formatting
    plt.title(f"Rolling SMAPE (Window: {window_size} periods)", fontsize=10)
    plt.ylabel("SMAPE (%)")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add horizontal line at the overall average SMAPE for reference
    for strategy_name, prediction in results.items():
        overall_smape = (2 * np.abs(prediction['Actual'] - prediction['Predicted']) /
                       (np.abs(prediction['Actual']) + np.abs(prediction['Predicted']))).mean() * 100
        plt.axhline(y=overall_smape, linestyle='--', alpha=0.5, 
                   color=plt.gca().lines[-1].get_color())
    
    plt.tight_layout()
    plt.show()

def plot_regime_performance_time(results, window_size=22, is_intraday=True):
    vol_data = results[list(results.keys())[0]]['Actual']
    if is_intraday:
        low, high = vol_data.quantile(0.5), vol_data.quantile(0.9)
    else:
        low, high = vol_data.quantile(0.33), vol_data.quantile(0.67)
    
    low_mask = vol_data <= low
    medium_mask = (vol_data > low) & (vol_data <= high)
    high_mask = vol_data > high
    
    _, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    regimes = ['Low Volatility', 'Medium Volatility', 'High Volatility']
    masks = [low_mask, medium_mask, high_mask]
    
    # Plot each regime
    for i, (regime, mask) in enumerate(zip(regimes, masks)):
        ax = axes[i]
        
        for strategy_name, prediction in results.items():
            regime_data = prediction[mask].copy()
            if len(regime_data) > 0:
                # Calculate rolling SMAPE
                ape = (2 * np.abs(regime_data['Actual'] - regime_data['Predicted']) /
                      (np.abs(regime_data['Actual']) + np.abs(regime_data['Predicted'])))
                
                if sum(mask) < 100:  # If sparse data
                    # Group by day or hour depending on data type
                    if is_intraday:
                        grouped = ape.groupby(pd.Grouper(freq='H')).mean() * 100
                    else:
                        grouped = ape.groupby(pd.Grouper(freq='D')).mean() * 100
                    ax.plot(grouped.index, grouped, label=strategy_name)
                else:
                    # Regular rolling window for dense data
                    rolling_smape = ape.rolling(window=window_size).mean() * 100
                    ax.plot(regime_data.index, rolling_smape, label=strategy_name)
                
                avg_smape = ape.mean() * 100
                ax.axhline(y=avg_smape, linestyle='--', alpha=0.5, 
                           color=ax.lines[-1].get_color())
        
        ax.set_title(f"{regime} (N={sum(mask)} points)")
        ax.set_ylabel("SMAPE (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()
