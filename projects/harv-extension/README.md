# Enhancing the HAR-VR Models for Volatility Prediction

## **Motivation**
Volatility prediction is critical for financial analysis, portfolio management, and risk assessment. Traditional HAR-RV (Heterogeneous Autoregressive Realized Volatility) models provide a foundation for predicting realized volatility using daily, weekly, and monthly averages of squared returns. However, this project aims to explore extended modeling strategies that incorporate innovative feature engineering techniques such as exhaustive search, Hamming codes, and prime modulo classes. These extensions aim to improve predictive performance by capturing diverse patterns and periodicities in financial time series data.

## **Project Overview**
This project extends the standard HAR-RV model by introducing advanced feature engineering strategies and benchmarking their predictive performance. The workflow includes:

1. **Data Collection:** Fetch financial data from Yahoo Finance using `yfinance`.
2. **Feature Engineering:** Create volatility-based features using innovative strategies.
3. **Model Training and Prediction:** Use Ordinary Least Squares (OLS) regression to predict daily realized volatility.
4. **Performance Evaluation:** Compare strategies using metrics like SMAPE (Symmetric Mean Absolute Percentage Error).
5. **Visualization:** Generate plots to compare predicted and actual volatility for all strategies.

## **Strategies**
1. **Standard HAR-RV Model:**
   - Incorporates traditional daily, weekly, and monthly rolling averages of squared returns.
   
2. **Exhaustive Search:**
   - Creates features by computing rolling averages over all possible window sizes up to a specified limit.

3. **Hamming Codes:**
   - Encodes time indices into binary representation, creating features by weighting squared returns based on binary patterns.

4. **Prime Modulo Classes:**
   - Uses modular arithmetic with prime numbers to create features that highlight periodic patterns in volatility.

## **Thought Process**
- The HAR-RV model captures hierarchical dependencies in volatility, but it might miss complex patterns. 
- Exhaustive search provides a brute-force way to explore all possible dependencies.
- Hamming codes offer a compact, systematic way to encode temporal dependencies, inspired by error-correcting codes.
- Prime modulo classes introduce mathematical structure to identify cyclical trends often present in market data.
- By combining these diverse approaches, the goal is to maximize predictive accuracy and reveal insights into volatility behavior.

## **Setup Instructions**
### Prerequisites
- Python 3.11 environment (using Conda).
- Dependencies listed in `requirements.txt`.

### Install Dependencies
1. Create and activate a Conda environment:
   ```bash
   conda create -n volatility_env python=3.11
   conda activate volatility_env
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Run the Project
To run the main comparison script:
```bash
python script_name.py
```
Replace `script_name.py` with the name of your script file.

### Example Output
The script fetches stock data for Apple (`AAPL`) from January 2020 to January 2024 and compares the performance of all strategies. Results include SMAPE scores for each strategy and plots showing actual vs. predicted volatility.

## **Directory Structure**
- `script_name.py`: Main script implementing the project.
- `requirements.txt`: Contains Python package dependencies.
- `README.md`: This file.

## **Performance Metrics**
- **SMAPE**: Used to evaluate prediction accuracy. Lower SMAPE indicates better performance.
- **Visual Comparison**: Plots help assess temporal alignment of predicted and actual volatility.

## **Future Work**
- Extend feature engineering with Fourier transformations or machine learning-based approaches.
- Test on additional tickers and asset classes to generalize findings.
- Experiment with advanced models like LSTMs or Transformer architectures.

## **Acknowledgments**
This project leverages open-source tools and libraries such as `yfinance`, `numpy`, `pandas`, `statsmodels`, and `matplotlib`. Special thanks to the financial modeling community for inspiring the exploration of advanced volatility prediction techniques.

---

Enjoy exploring innovative ways to predict market volatility!