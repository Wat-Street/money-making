import unittest
import pandas as pd
import numpy as np
from feature_transform import FeatureTransform

class TestFeatureTransform(unittest.TestCase):

    """
    Run All Tests in a Group (for a particular function e.g. rolling_beta from feature_transform.py):
        python test_feature_transform.py rolling_beta

    The above will run all tests (valid, edge, and invalid) for the rolling_beta group.

    Run Only a Specific Category of Tests in a Group:
        To run only valid input tests for rolling_beta:
            python test_feature_transform.py rolling_beta valid

        To run only edge case tests for rolling_beta:
            python test_feature_transform.py rolling_beta edge

        To run only invalid input tests for rolling_beta:
            python test_feature_transform.py rolling_beta invalid

    Run All Tests for All Groups:
        python test_feature_transform.py
    """

    @classmethod
    def setUpClass(cls):
        # Load sample data from the CSV file
        cls.data = pd.read_csv("sample_data.csv")
        cls.data['date'] = pd.to_datetime(cls.data['date'])

        # Assuming 'month_return' column is used for both security and benchmark for testing
        cls.security_series = cls.data['month_return']
        cls.benchmark_series = cls.data['month_return']

    ############################################
    # Tests for rolling_beta
    ############################################

    # Test valid input for rolling_beta
    def test_rolling_beta_valid_input(self):
        window = 20
        min_window_pct = 0.8

        beta_series = FeatureTransform.rolling_beta(self.security_series, self.benchmark_series, window, min_window_pct)
        
        # Check the length of beta_series is the same as the input series
        self.assertEqual(len(beta_series), len(self.security_series))

        # Check beta_series contains numeric values
        self.assertTrue(np.issubdtype(beta_series.dtype, np.number))

    # Test edge cases for rolling_beta
    def test_rolling_beta_edge_cases(self):
        # Test with small window
        beta_series_small_window = FeatureTransform.rolling_beta(self.security_series, self.benchmark_series, 1, 0.8)
        self.assertEqual(len(beta_series_small_window), len(self.security_series))

        # Test with large window
        beta_series_large_window = FeatureTransform.rolling_beta(self.security_series, self.benchmark_series, 1000, 0.8)
        self.assertEqual(len(beta_series_large_window), len(self.security_series))

        # Test with min_window_pct at 0 and 1
        beta_series_min_pct = FeatureTransform.rolling_beta(self.security_series, self.benchmark_series, 20, 0)
        beta_series_max_pct = FeatureTransform.rolling_beta(self.security_series, self.benchmark_series, 20, 1)
        self.assertEqual(len(beta_series_min_pct), len(self.security_series))
        self.assertEqual(len(beta_series_max_pct), len(self.security_series))

    # Test invalid input for rolling_beta
    def test_rolling_beta_invalid_input(self):
        # Test with invalid window size
        with self.assertRaises(AssertionError):
            FeatureTransform.rolling_beta(self.security_series, self.benchmark_series, -1, 0.8)

        # Test with invalid min_window_pct
        with self.assertRaises(AssertionError):
            FeatureTransform.rolling_beta(self.security_series, self.benchmark_series, 20, -0.1)

        with self.assertRaises(AssertionError):
            FeatureTransform.rolling_beta(self.security_series, self.benchmark_series, 20, 1.1)
    
    ############################################
    # Tests for percent_from_trailing_max
    ############################################

    # Test valid input for percent_from_trailing_max
    def test_trailing_max_valid_input(self):
        window = 20
        min_window_pct = 0.8
        result_series = FeatureTransform.percent_from_trailing_max(self.sample_series, window, min_window_pct)

        self.assertEqual(len(result_series), len(self.sample_series))
        self.assertTrue(np.issubdtype(result_series.dtype, np.number))

    # Test edge cases for percent_from_trailing_max
    def test_trailing_max_edge_cases(self):
        # Test with small window
        result_series_small = FeatureTransform.percent_from_trailing_max(self.sample_series, 1, 0.8)
        self.assertEqual(len(result_series_small), len(self.sample_series))

        # Test with large window
        result_series_large = FeatureTransform.percent_from_trailing_max(self.sample_series, 1000, 0.8)
        self.assertEqual(len(result_series_large), len(self.sample_series))

        # Test with min_window_pct at 0 and 1
        result_series_min_pct = FeatureTransform.percent_from_trailing_max(self.sample_series, 20, 0)
        result_series_max_pct = FeatureTransform.percent_from_trailing_max(self.sample_series, 20, 1)
        self.assertEqual(len(result_series_min_pct), len(self.sample_series))
        self.assertEqual(len(result_series_max_pct), len(self.sample_series))

    # Test invalid input for percent_from_trailing_max
    def test_trailing_max_invalid_input(self):
        # Test with invalid window size
        with self.assertRaises(AssertionError):
            FeatureTransform.percent_from_trailing_max(self.sample_series, -1, 0.8)

        # Test with invalid min_window_pct
        with self.assertRaises(AssertionError):
            FeatureTransform.percent_from_trailing_max(self.sample_series, 20, -0.1)
        with self.assertRaises(AssertionError):
            FeatureTransform.percent_from_trailing_max(self.sample_series, 20, 1.1)

    ############################################
    # Tests for percent_from_trailing_min
    ############################################

    # Test valid input for percent_from_trailing_min
    def test_trailing_min_valid_input(self):
        window = 20
        min_window_pct = 0.8
        result_series = FeatureTransform.percent_from_trailing_min(self.sample_series, window, min_window_pct)

        self.assertEqual(len(result_series), len(self.sample_series))
        self.assertTrue(np.issubdtype(result_series.dtype, np.number))

    # Test edge cases for percent_from_trailing_min
    def test_trailing_min_edge_cases(self):
        # Test with small window
        result_series_small = FeatureTransform.percent_from_trailing_min(self.sample_series, 1, 0.8)
        self.assertEqual(len(result_series_small), len(self.sample_series))

        # Test with large window
        result_series_large = FeatureTransform.percent_from_trailing_min(self.sample_series, 1000, 0.8)
        self.assertEqual(len(result_series_large), len(self.sample_series))

        # Test with min_window_pct at 0 and 1
        result_series_min_pct = FeatureTransform.percent_from_trailing_min(self.sample_series, 20, 0)
        result_series_max_pct = FeatureTransform.percent_from_trailing_min(self.sample_series, 20, 1)
        self.assertEqual(len(result_series_min_pct), len(self.sample_series))
        self.assertEqual(len(result_series_max_pct), len(self.sample_series))

    # Test invalid input for percent_from_trailing_min
    def test_trailing_min_invalid_input(self):
        # Test with invalid window size
        with self.assertRaises(AssertionError):
            FeatureTransform.percent_from_trailing_min(self.sample_series, -1, 0.8)

        # Test with invalid min_window_pct
        with self.assertRaises(AssertionError):
            FeatureTransform.percent_from_trailing_min(self.sample_series, 20, -0.1)
        with self.assertRaises(AssertionError):
            FeatureTransform.percent_from_trailing_min(self.sample_series, 20, 1.1)
    
    ############################################
    # Tests for sma
    ############################################

    # Test for sma with valid input
    def test_sma_valid_input(self):
        window = 20
        min_window_pct = 0.8
        result_series = FeatureTransform.sma(self.sample_series, window, min_window_pct)

        self.assertEqual(len(result_series), len(self.sample_series))
        self.assertTrue(np.issubdtype(result_series.dtype, np.number))

    # Test edge cases for sma
    def test_sma_edge_cases(self):
        # Test with small window
        result_series_small = FeatureTransform.sma(self.sample_series, 1, 0.8)
        self.assertEqual(len(result_series_small), len(self.sample_series))

        # Test with large window
        result_series_large = FeatureTransform.sma(self.sample_series, 1000, 0.8)
        self.assertEqual(len(result_series_large), len(self.sample_series))

        # Test with min_window_pct at 0 and 1
        result_series_min_pct = FeatureTransform.sma(self.sample_series, 20, 0)
        result_series_max_pct = FeatureTransform.sma(self.sample_series, 20, 1)
        self.assertEqual(len(result_series_min_pct), len(self.sample_series))
        self.assertEqual(len(result_series_max_pct), len(self.sample_series))

    # Test invalid input for sma
    def test_sma_invalid_input(self):
        # Test with invalid window size
        with self.assertRaises(AssertionError):
            FeatureTransform.sma(self.sample_series, -1, 0.8)

        # Test with invalid min_window_pct
        with self.assertRaises(AssertionError):
            FeatureTransform.sma(self.sample_series, 20, -0.1)
        with self.assertRaises(AssertionError):
            FeatureTransform.sma(self.sample_series, 20, 1.1)

    ############################################
    # Tests for ema
    ############################################

    # Test for ema with valid input
    def test_ema_valid_input(self):
        window = 20
        min_window_pct = 0.8
        result_series = FeatureTransform.ema(self.sample_series, window, min_window_pct)

        self.assertEqual(len(result_series), len(self.sample_series))
        self.assertTrue(np.issubdtype(result_series.dtype, np.number))

        # Optionally test with different 'adjust', 'halflife', 'span', 'com' parameters

    # Test edge cases for ema
    def test_ema_edge_cases(self):
        # Test with small window
        result_series_small = FeatureTransform.ema(self.sample_series, 1, 0.8)
        self.assertEqual(len(result_series_small), len(self.sample_series))

        # Test with large window
        result_series_large = FeatureTransform.ema(self.sample_series, 1000, 0.8)
        self.assertEqual(len(result_series_large), len(self.sample_series))

        # Test with min_window_pct at 0 and 1
        result_series_min_pct = FeatureTransform.ema(self.sample_series, 20, 0)
        result_series_max_pct = FeatureTransform.ema(self.sample_series, 20, 1)
        self.assertEqual(len(result_series_min_pct), len(self.sample_series))
        self.assertEqual(len(result_series_max_pct), len(self.sample_series))

    # Test invalid input for ema
    def test_ema_invalid_input(self):
        # Test with invalid window size
        with self.assertRaises(AssertionError):
            FeatureTransform.ema(self.sample_series, -1, 0.8)

        # Test with invalid min_window_pct
        with self.assertRaises(AssertionError):
            FeatureTransform.ema(self.sample_series, 20, -0.1)
        with self.assertRaises(AssertionError):
            FeatureTransform.ema(self.sample_series, 20, 1.1)


if __name__ == '__main__':
    suite = unittest.TestSuite()

    # For better modularity later on; contributors can add test of each category
    test_groups = {
        'rolling_beta': {
            'valid': ['test_rolling_beta_valid_input'],
            'edge': ['test_rolling_beta_edge_cases'],
            'invalid': ['test_rolling_beta_invalid_input']
        },
        'trailing_max': {
            'valid': ['test_trailing_max_valid_input'],
            'edge': ['test_trailing_max_edge_cases'],
            'invalid': ['test_trailing_max_invalid_input']
        },
        'trailing_min': {
            'valid': ['test_trailing_min_valid_input'],
            'edge': ['test_trailing_min_edge_cases'],
            'invalid': ['test_trailing_min_invalid_input']
        },
        'sma': {
            'valid': ['test_sma_valid_input'],
            'edge': ['test_sma_edge_cases'],
            'invalid': ['test_sma_invalid_input']
        },
        'ema': {
            'valid': ['test_ema_valid_input'],
            'edge': ['test_ema_edge_cases'],
            'invalid': ['test_ema_invalid_input']
        }
    }

    if len(sys.argv) > 1:
        group_name = sys.argv[1]
        if group_name in test_groups:
            for test in test_groups[group_name]:
                suite.addTest(TestFeatureTransform(test))
        else:
            print(f"Unknown test group: {group_name}")
            sys.exit(1)
    else:
        for group in test_groups.values():
            for test in group:
                suite.addTest(TestFeatureTransform(test))

    runner = unittest.TextTestRunner()
    runner.run(suite)