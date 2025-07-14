import pytest
import pandas as pd
from unittest.mock import patch
from analysis.data_loader import get_stock_data

@patch("os.path.exists", return_value=True)
@patch("pandas.read_csv")

def test_get_stock_data(mock_read_csv, mock_exists):
    # mock the data
    mock_df = pd.DataFrame({
        "Date": ["2025-07-01"],
        "Close": [150.0],
        "Ticker": ["AAPL"]
    })

    mock_read_csv.return_value = mock_df

    # use function, should be the same as the mock

    result = get_stock_data("AAPL", "2025-07-01", "2025-07-07", cache=True)

    mock_exists.assert_called_once()
    mock_read_csv.assert_called_once_with("data/raw/AAPL_2025-07-01_to_2025-07-07_raw.csv", parse_dates=["Date"])
    pd.testing.assert_frame_equal(result, mock_df)