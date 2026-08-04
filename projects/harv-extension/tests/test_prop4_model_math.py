from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "outputs" / "cp_repo_ops_hg_tests" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_cp_repo_ops_hg_tests as runner  # noqa: E402
import validate_pm_native_geometry as validation  # noqa: E402


class Prop4ModelMathTests(unittest.TestCase):
    def test_prop4_registry_has_exactly_eighteen_models(self) -> None:
        registry = runner.build_model_registry(22)["models"]
        self.assertEqual(len(runner.PROP4_MODELS), 18)
        self.assertEqual(runner.PROP4_MODELS[-1], "CP_REPO_OPS_K")
        self.assertTrue(set(runner.PROP4_MODELS).issubset(registry))

    def test_cp_quotient_projection_is_orthogonal(self) -> None:
        c_mat, m_c = runner.cp_orthogonal_projection(22)
        np.testing.assert_allclose(m_c, m_c.T, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(m_c @ m_c, m_c, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(c_mat @ m_c, 0.0, atol=1e-12, rtol=0.0)

    def test_incremental_qr_matches_direct_rank_deficient_least_squares(self) -> None:
        rng = np.random.default_rng(42)
        base = rng.normal(size=(800, 5))
        x = np.column_stack([np.ones(len(base)), base, base[:, 0] + base[:, 1]])
        y = x @ np.array([0.3, 0.2, -0.1, 0.05, 0.4, -0.2, 0.1]) + rng.normal(scale=0.01, size=len(x))
        r_mat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
        qty = np.zeros(x.shape[1], dtype=float)
        for row, target in zip(x, y):
            runner.qr_insert_observation(r_mat, qty, row, target)
        beta_fast = np.linalg.lstsq(r_mat, qty, rcond=None)[0]
        beta_direct = np.linalg.lstsq(x, y, rcond=None)[0]
        test_base = rng.normal(size=(100, 5))
        test_x = np.column_stack([np.ones(len(test_base)), test_base, test_base[:, 0] + test_base[:, 1]])
        np.testing.assert_allclose(test_x @ beta_fast, test_x @ beta_direct, atol=1e-11, rtol=1e-11)

    def test_causal_random_gate_only_samples_prior_rows(self) -> None:
        rows = 140
        values = 0.2 + 0.01 * np.sin(np.arange(rows) / 7.0)
        frame = pd.DataFrame({"RV_d": values})
        for lag in range(1, 23):
            frame[f"lag{lag}"] = frame["RV_d"].shift(lag).bfill()
        gated = runner.add_gate_columns(frame, 22, runner.fallback_blocks(22), 345)
        source = gated["GATE_RANDOM_SOURCE_INDEX"].to_numpy(dtype=int)
        real = gated["GATE_REAL"].to_numpy(dtype=float)
        random = gated["GATE_RANDOM"].to_numpy(dtype=float)
        self.assertEqual(source[0], -1)
        self.assertTrue(np.all(source[1:] < np.arange(1, rows)))
        np.testing.assert_allclose(random[1:], real[source[1:]], atol=0.0, rtol=0.0)

    def test_phqo_falls_back_to_exact_cp_on_zero_paths(self) -> None:
        rows = 50
        index = pd.date_range("2024-01-01", periods=rows, freq="5min")
        frame = pd.DataFrame({"RV_d": np.zeros(rows)}, index=index)
        for lag in range(1, 23):
            frame[f"lag{lag}"] = 0.0
        args = SimpleNamespace(n=22, pm_low_modes=12)
        cp_map = {pd.Timestamp(index[i + 1]): 0.00123 for i in range(22, rows - 1)}
        with mock.patch.object(runner, "cp_prediction_map", return_value=cp_map):
            values = runner.phqo_predict_values(
                frame,
                {"CP_REPO_FRESH": []},
                args,
                {"tau": 0.1, "gamma": 2.0},
                22,
                rows - 1,
            )
        self.assertEqual(len(values), rows - 23)
        np.testing.assert_allclose([value for _, value in values], 0.00123, atol=0.0, rtol=0.0)

    def test_pooled_rmse_is_root_of_pooled_mse(self) -> None:
        metric = pd.DataFrame(
            [
                {
                    "asset": "A",
                    "model_name": "M",
                    "benchmark_model": "CP_REPO_FRESH",
                    "subset": "all_observations",
                    "loss_metric": "RMSE",
                    "n_obs": 1,
                    "cp_loss": 1.0,
                    "model_loss": 2.0,
                    "advantage_cp_minus_model": -1.0,
                    "win_rate_vs_cp": 0.0,
                    "mean_row_loss_diff_cp_minus_model": -1.0,
                },
                {
                    "asset": "B",
                    "model_name": "M",
                    "benchmark_model": "CP_REPO_FRESH",
                    "subset": "all_observations",
                    "loss_metric": "RMSE",
                    "n_obs": 3,
                    "cp_loss": 3.0,
                    "model_loss": 4.0,
                    "advantage_cp_minus_model": -1.0,
                    "win_rate_vs_cp": 0.0,
                    "mean_row_loss_diff_cp_minus_model": -1.0,
                },
            ]
        )
        pooled = validation.aggregate_metric_rows(metric)
        row = pooled.loc[pooled["asset"] == "POOLED"].iloc[0]
        self.assertAlmostEqual(float(row["cp_loss"]), math.sqrt(7.0))
        self.assertAlmostEqual(float(row["model_loss"]), math.sqrt(13.0))

    def test_prop4_loss_aggregation_uses_true_pooled_rmse(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "asset": "A",
                    "model_name": "M",
                    "condition": "all_observations",
                    "condition_label_type": "not_conditioned",
                    "loss_variant": "standard",
                    "loss_metric": "RMSE",
                    "n_obs": 1,
                    "CP_loss": 1.0,
                    "model_loss": 2.0,
                    "advantage_CP_minus_model": -1.0,
                },
                {
                    "asset": "B",
                    "model_name": "M",
                    "condition": "all_observations",
                    "condition_label_type": "not_conditioned",
                    "loss_variant": "standard",
                    "loss_metric": "RMSE",
                    "n_obs": 3,
                    "CP_loss": 3.0,
                    "model_loss": 4.0,
                    "advantage_CP_minus_model": -1.0,
                },
            ]
        )
        row = runner.aggregate_alternative_loss(frame).iloc[0]
        self.assertAlmostEqual(float(row["pooled_CP_loss"]), math.sqrt(7.0))
        self.assertAlmostEqual(float(row["pooled_model_loss"]), math.sqrt(13.0))

    def test_exact_pooling_matches_concatenated_row_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            pred_dir = outdir / "predictions" / "full"
            pred_dir.mkdir(parents=True)
            expected_advantages = []
            for offset, asset in enumerate(["A", "B"]):
                actual = np.linspace(0.001 + offset * 0.0002, 0.003 + offset * 0.0002, 40)
                cp_pred = actual * (1.06 - 0.01 * offset)
                model_pred = actual * (0.96 + 0.015 * np.sin(np.arange(40) / 3.0 + offset))
                frame = pd.DataFrame(
                    {
                        "Date": pd.date_range("2024-01-01", periods=40, freq="5min"),
                        "Actual": actual,
                        "Predicted_CP_REPO_FRESH": cp_pred,
                        "Predicted_TEST_MODEL": model_pred,
                    }
                )
                frame.to_csv(pred_dir / f"{asset}.csv", index=False)
                expected_advantages.extend(
                    (
                        runner.smape(pd.Series(actual[22:]), pd.Series(cp_pred[22:]))
                        - runner.smape(pd.Series(actual[22:]), pd.Series(model_pred[22:]))
                    ).tolist()
                )

            results = runner.compute_results(outdir, "full", ["A", "B"], ["TEST_MODEL"], 22)
            row = results["pooled"].loc[
                (results["pooled"]["model_name"] == "TEST_MODEL")
                & (results["pooled"]["condition"] == "all_observations")
            ].iloc[0]
            self.assertEqual(int(row["n_obs"]), len(expected_advantages))
            self.assertAlmostEqual(float(row["median_advantage_vs_CP"]), float(np.median(expected_advantages)))


if __name__ == "__main__":
    unittest.main()
