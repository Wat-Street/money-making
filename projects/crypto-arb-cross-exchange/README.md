# Crypto Arbitrage Project

## Summary

This case study explores arbitrage opportunities between cryptocurrency exchanges
Poloniex and Binance. We focus on the analysis of trade-level data from the given dataset
(Aug 2022). The goal is to identify price differentials and evaluate the potential
profitability of arbitrage strategies. The study analyzes various indicators including value and duration of significant differentials, driving factors, alternative exchanges, illiquidity, and limitations of the data. Findings reveal that significant arbitrage opportunities are sporadic and short-lived, meaning smart, low-latency programs must be used for profitable execution.

## File Layout

- `arbitrage/` contains the main code + Jupyter notebook to test for arbitrage opportuniteis across the two exchanges based on the data
- `data_ingestion/` contains all code related to the data loading, preprocessing, and more
- `model/` contains the code used to create the time series model
