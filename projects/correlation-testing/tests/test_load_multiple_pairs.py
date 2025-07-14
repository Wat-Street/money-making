import pytest
import pandas as pd
from unittest.mock import patch
from analysis.data_loader import load_multiple_pairs

  

@patch("analysis.data_loader.load_pair_data")
def test_load_multiple_pairs(mock_load_pair_data):

    # Mock data for load_pair_data
    mock_load_pair_data.side_effect = lambda ticker_a, ticker_b, start_date, end_date, cache: pd.DataFrame({
        'date': ['2025-07-01', '2025-07-02'],
        f'{ticker_a}_price': [100, 101],
        f'{ticker_b}_price': [200, 201]
    })

    # input
    pairs = [('AAPL', 'MSFT'), ('GOOG', 'AMZN')]
    start_date = '2025-07-01'
    end_date = '2025-07-02'
    cache = True

    # expected
    expected_output = {
        ('AAPL', 'MSFT'): pd.DataFrame({
        'date': ['2025-07-01', '2025-07-02'],
        'AAPL_price': [100, 101],
        'MSFT_price': [200, 201]
    }),
        ('GOOG', 'AMZN'): pd.DataFrame({
        'date': ['2025-07-01', '2025-07-02'],
        'GOOG_price': [100, 101],
        'AMZN_price': [200, 201]
    })
    }

    result = load_multiple_pairs(pairs, start_date, end_date, cache)

    # assert
    for pair in pairs:
        pd.testing.assert_frame_equal(result[pair], expected_output[pair])