
import os
import pytest

from analysis.data_loader import get_data_path

# note: this test will actually make a file at data/raw/[whatever name].csv
def test_get_data_path_default_folder():
    ticker = "AAPL"
    start_date = "2025-01-01"
    end_date = "2025-12-31"
    expected_path = os.path.join("data/raw", "AAPL_2025-01-01_to_2025-12-31_raw.csv")
    assert get_data_path(ticker, start_date, end_date) == expected_path

# note: this test will actually make a file at data/custom/[whatever name].csv
def test_get_data_path_custom_folder():
    ticker = "GOOGL"
    start_date = "2025-01-01"
    end_date = "2025-12-31"

    folder = "data/custom"
    expected_path = os.path.join(folder, "GOOGL_2025-01-01_to_2025-12-31_raw.csv")
    assert get_data_path(ticker, start_date, end_date, folder) == expected_path