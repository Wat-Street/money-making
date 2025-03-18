import numpy as np
import matplotlib.pyplot as plt

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
