import pytest
import pandas as pd
from unittest.mock import patch
from analysis.data_loader import load_pair_data
from analysis.data_loader import get_stock_data

@patch("analysis.data_loader.get_stock_data")
def test_load_pair_data_success(mock_get_stock_data):
    mock_data_a = pd.DataFrame({
        "Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "Close": [150, 152, 154],
        "Ticker": ["AAPL", "AAPL", "AAPL"]
    })

    mock_data_b = pd.DataFrame({
        "Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "Close": [300, 305, 310],
        "Ticker": ["MSFT", "MSFT", "MSFT"]
    })

    mock_get_stock_data.side_effect = [mock_data_a, mock_data_b]
    result = load_pair_data("AAPL", "MSFT", "2025-01-01", "2025-01-03")

    expected = pd.DataFrame({
        "Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "Close_AAPL": [150, 152, 154],
        "Ticker_AAPL": ["AAPL", "AAPL", "AAPL"],
        "Close_MSFT": [300, 305, 310],
        "Ticker_MSFT": ["MSFT", "MSFT", "MSFT"]
    })

    pd.testing.assert_frame_equal(result, expected)

# testing the part where it only returns aligned dates
@patch("analysis.data_loader.get_stock_data")
def test_load_pair_data_mismatched_dates(mock_get_stock_data):
    mock_data_a_mismatched = pd.DataFrame({
        "Date": ["2025-01-01", "2025-01-02"],
        "Close": [150, 152],
        "Ticker": ["AAPL", "AAPL"]
    })

    mock_data_b_mismatched = pd.DataFrame({
        "Date": ["2025-01-02", "2025-01-03"],
        "Close": [305, 310],
        "Ticker": ["MSFT", "MSFT"]
    })

    mock_get_stock_data.side_effect = [mock_data_a_mismatched, mock_data_b_mismatched]
    result = load_pair_data("AAPL", "MSFT", "2025-01-01", "2025-01-03")

    # only return the entries that are in the union of the 2 data, so only 2025-01-02 counts
    expected = pd.DataFrame({
        "Date": ["2025-01-02"],
        "Close_AAPL": [152],
        "Ticker_AAPL": ["AAPL"],
        "Close_MSFT": [305],
        "Ticker_MSFT": ["MSFT"]
    })
    pd.testing.assert_frame_equal(result, expected)