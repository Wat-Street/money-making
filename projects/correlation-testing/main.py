# This file contains skeleton code for now to show how compute_lagged_correlation should be utilized

# from analysis.data_loader import load_pair_data
# from analysis.pairs import research_pairs
# from analysis.correlation import compute_lagged_correlation

# """
# Sample Output
# CVS-JNJ Lagged Correlation:
#     lag  correlation
# 0   -10     0.4512
# 1    -9     0.4620
# ...
# 10    0     0.4925
# ...
# 20   10     0.4458
# """
# for stock_a, stock_b in research_pairs:
#     data = load_pair_data(stock_a, stock_b, "2022-01-01", "2024-12-31")

#     corr_df = compute_lagged_correlation(
#         data[f"Close_{stock_a}"],
#         data[f"Close_{stock_b}"],
#         max_lag=10
#     )

#     print(f"\n{stock_a}-{stock_b} Lagged Correlation:")
#     print(corr_df)