from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "outputs" / "pm_cast_d_warmup600" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pm_cast_d as cast  # noqa: E402


class PMCastDMathTests(unittest.TestCase):
    def test_warmup_600_is_default_and_required(self) -> None:
        args = cast.build_parser().parse_args(["--mode", "tune"])
        self.assertEqual(args.warmup, 600)

    def test_pm_and_matched_operators_annihilate_cp_space(self) -> None:
        operators = {}
        for family in ["pm", "spectral_random", "contiguous"]:
            operator, c_mat = cast.operator_for_family(family, 22, 0.5, 12, 42)
            operators[family] = operator
            np.testing.assert_allclose(operator @ c_mat.T, 0.0, atol=1e-11, rtol=0.0)
        pm_singular = np.linalg.svd(operators["pm"], compute_uv=False)
        for family in ["spectral_random", "contiguous"]:
            np.testing.assert_allclose(
                np.linalg.svd(operators[family], compute_uv=False), pm_singular, atol=1e-11, rtol=1e-11
            )

    def test_transport_weights_are_positive_and_sum_to_one(self) -> None:
        rng = np.random.default_rng(42)
        x = np.exp(rng.normal(-7.0, 0.8, size=(100, 22)))
        operator, _ = cast.operator_for_family("pm", 22, 1.0, 12, 42)
        ratio, audit = cast.transport_log_ratio(x, operator, 0.12, -0.5, return_weight_audit=True)
        self.assertTrue(np.isfinite(ratio).all())
        self.assertGreater(audit["minimum_weight"], 0.0)
        self.assertLessEqual(audit["max_abs_weight_sum_error"], 1e-12)

    def test_scalar_and_vectorized_transport_are_identical(self) -> None:
        rng = np.random.default_rng(7)
        x = np.exp(rng.normal(-7.0, 0.8, size=(25, 22)))
        operator, _ = cast.operator_for_family("pm", 22, 0.5, 12, 42)
        vector, _ = cast.transport_log_ratio(x, operator, 0.07, -0.25)
        scalar = np.array([cast.scalar_transport_log_ratio(row, operator, 0.07, -0.25) for row in x])
        np.testing.assert_allclose(vector, scalar, atol=1e-14, rtol=1e-14)

    def test_zero_action_nests_cp_exactly(self) -> None:
        cp = np.array([-0.01, 0.0, 0.1, 0.2])
        ratio = np.array([1.0, -1.0, 0.5, -0.5])
        ranks = np.array([0.2, 0.4, 0.8, 0.95])
        pred, correction = cast.apply_action(
            cp, ratio, ranks, {"intercept": 0.0, "alpha": 0.0, "clip": 0.1, "tail_rule": "q90_zero"}
        )
        np.testing.assert_array_equal(pred, cp)
        np.testing.assert_array_equal(correction, np.zeros_like(cp))

    def test_expanding_rank_uses_strictly_prior_values(self) -> None:
        values = np.array([3.0, 1.0, 2.0, 4.0])
        original = cast.expanding_prior_rank(values)
        changed = cast.expanding_prior_rank(np.array([3.0, 1.0, 2.0, 1e12]))
        np.testing.assert_array_equal(original[:3], changed[:3])
        self.assertAlmostEqual(original[2], 0.5)

    def test_pooled_metrics_and_block_bootstrap_are_exact_and_finite(self) -> None:
        rows = []
        for asset_index, asset in enumerate(["A", "B"]):
            for idx in range(20):
                actual = 1.0 + 0.01 * idx + 0.1 * asset_index
                cp = actual + 0.10
                row = {
                    "asset": asset,
                    "Date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=idx // 2, minutes=idx),
                    "Actual": actual,
                    "Predicted_CP_REPO_FRESH": cp,
                }
                for model in cast.MODEL_FAMILIES:
                    row[f"Predicted_{model}"] = actual + 0.05
                row["Predicted_PM_CAST_D_Q05"] = actual - 0.10
                row["Predicted_PM_CAST_D_Q50"] = actual
                row["Predicted_PM_CAST_D_Q95"] = actual + 0.10
                rows.append(row)
        panel = pd.DataFrame(rows)
        metrics, consistency, comparison = cast.metric_tables(panel)
        pooled = metrics[(metrics["asset"] == "POOLED") & (metrics["model_name"] == "PM_CAST_D")].set_index("loss_metric")
        self.assertAlmostEqual(float(pooled.loc["MAE", "CP_loss"]), 0.10)
        self.assertAlmostEqual(float(pooled.loc["MAE", "model_loss"]), 0.05)
        self.assertAlmostEqual(float(pooled.loc["RMSE", "CP_loss"]), 0.10)
        self.assertAlmostEqual(float(pooled.loc["RMSE", "model_loss"]), 0.05)
        self.assertTrue((consistency[consistency["model_name"] == "PM_CAST_D"]["assets_positive"] == 2).all())
        self.assertTrue(comparison["PM_beats_control"].eq(False).all())
        distribution = cast.distribution_table(panel).set_index("asset")
        self.assertAlmostEqual(float(distribution.loc["POOLED", "empirical_coverage"]), 1.0)
        bootstrap = cast.moving_block_bootstrap(panel, "PM_CAST_D", reps=30, block_days=2, seed=42)
        self.assertEqual(set(bootstrap["loss_metric"]), set(cast.METRICS))
        self.assertTrue(np.isfinite(bootstrap[["ci_low", "ci_high", "bootstrap_mean_advantage"]].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
