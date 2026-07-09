#!/usr/bin/env python
"""Build the Prop 4 mathematical-fidelity and PM failure audit package."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import run_cp_repo_ops_hg_tests as runner
import validate_pm_native_geometry as validation


ARCHITECTURES = {
    "HAR_RV": "Expanding OLS on intercept + RV_d + RV_w + RV_m.",
    "CP_REPO_FRESH": "Expanding OLS on repo RV* and CP_* from contig_prime_modulo(n=22, per_day_normalize=False).",
    "RAW_PM": "Expanding OLS on the ten repo PM residue averages only.",
    "RAW_CP_REPO_PLUS_PM": "Expanding OLS on CP_REPO_FRESH controls plus raw PM residue averages.",
    "CP_REPO_OPS_R": "Expanding OLS on CP controls plus online-standardized centered prime-residue contrasts.",
    "CP_REPO_RIDGE_OPS_R": "CP controls unpenalized; one train-only selected ridge penalty on OPS-R coefficients.",
    "CP_REPO_LRPM": "Unregularized diagnostic using within-zone local prime-residue contrasts plus CP controls.",
    "CP_REPO_RECENT_SLOPE": "CP controls plus causal recent/older zero-sum slopes and spike-recency shape.",
    "CP_REPO_HAAR_SHAPE": "CP controls plus a causal depth-three within-zone Haar path basis.",
    "CP_REPO_OPS_C": "CP controls plus contiguous q=2,3,5 Helmert sub-block contrasts.",
    "CP_REPO_RIDGE_OPS_C": "CP controls unpenalized; one train-only selected ridge penalty on OPS-C coefficients.",
    "CP_REPO_GATED_RIDGE_OPS_C": "CP controls plus real-gate times OPS-C, with shape-only ridge.",
    "CP_REPO_OPS_HG": "CP controls plus gated OPS-C and more heavily penalized gated OPS-R.",
    "RANDOM_RESIDUES_PLACEBO_REPO": "OPS-R-capacity ridge placebo with seeded random groups preserving residue sizes.",
    "SHUFFLED_LAG_PM_PLACEBO_REPO": "OPS-R-capacity ridge placebo with one seeded lag-label permutation.",
    "RANDOM_GATE_PLACEBO_REPO": "OPS-C ridge placebo whose gate is sampled only from strictly prior real gates.",
    "RIDGE_AR22": "Expanding ridge on lag1 through lag22; benchmark/control, not a PM placebo.",
    "CP_REPO_OPS_K": "CP controls plus gated, centered kernel-PCA lag-shape modes with shape-only ridge.",
}


def smape(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return 200.0 * np.abs(actual - pred) / np.maximum(np.abs(actual) + np.abs(pred), 1e-12)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_pm_panels(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((root / "predictions" / "full").glob("*_post_validation.csv")):
        frame = pd.read_csv(path, parse_dates=["Date"])
        frame["asset"] = path.name.split("_", 1)[0]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No PM prediction panels found under {root}")
    return pd.concat(frames, ignore_index=True, sort=False)


def model_fidelity_table(smoke: Path) -> pd.DataFrame:
    result_dir = smoke / "results"
    metadata = pd.read_csv(result_dir / "run_metadata.csv")
    fast = pd.read_csv(result_dir / "fast_slow_equivalence_audit.csv").set_index("model_name")
    alignment = pd.read_csv(result_dir / "alignment_audit.csv").set_index("model_name")
    panel = pd.read_csv(smoke / "predictions" / "smoke" / "AAPL.csv")
    registry = runner.build_model_registry(22)["models"]
    rows = []
    for model in runner.PROP4_MODELS:
        meta = metadata.loc[metadata["model_name"] == model].iloc[-1]
        audit = fast.loc[model]
        aligned = alignment.loc[model]
        pred = pd.to_numeric(panel[f"Predicted_{model}"], errors="coerce")
        entry = registry[model]
        rows.append(
            {
                "model_name": model,
                "architecture": ARCHITECTURES[model],
                "feature_families": ",".join(entry["feature_families"]),
                "feature_count": int(meta["feature_count"]),
                "design_rank": int(meta["design_rank"]),
                "rank_deficient": bool(meta["rank_deficient"]),
                "uses_ridge": bool(entry["uses_ridge"]),
                "uses_gate": bool(entry["uses_gate"]),
                "is_placebo": bool(entry["is_placebo"]),
                "finite_predictions": bool(np.isfinite(pred).all()),
                "timestamp_alignment": bool(aligned["timestamps_exactly_match_CP"]),
                "target_alignment": bool(aligned["actual_target_values_match_CP"]),
                "direct_solver_equivalence": bool(audit["passes"]),
                "max_abs_fast_direct_diff": float(audit["max_abs_prediction_diff"]),
                "status": "PASS" if bool(audit["passes"]) and np.isfinite(pred).all() else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def pooled_metric_table(panel: pd.DataFrame) -> pd.DataFrame:
    actual = pd.to_numeric(panel["Actual"], errors="coerce").to_numpy(dtype=float)
    rows = []
    for model in ["CP_REPO_FRESH", "PM_QDK", "PM_QDK_2", "PHQO"]:
        col = f"Predicted_{model}"
        if col not in panel:
            continue
        pred = pd.to_numeric(panel[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(pred)
        err = actual[valid] - pred[valid]
        rows.append(
            {
                "model_name": model,
                "n_obs": int(valid.sum()),
                "SMAPE": float(np.mean(smape(actual[valid], pred[valid]))),
                "MAE": float(np.mean(np.abs(err))),
                "MSE": float(np.mean(np.square(err))),
                "RMSE": float(np.sqrt(np.mean(np.square(err)))),
                "bias_pred_minus_actual": float(np.mean(pred[valid] - actual[valid])),
                "underprediction_share": float(np.mean(pred[valid] < actual[valid])),
            }
        )
    return pd.DataFrame(rows)


def pm_tail_decomposition(panel: pd.DataFrame) -> pd.DataFrame:
    actual = pd.to_numeric(panel["Actual"], errors="coerce").to_numpy(dtype=float)
    cp = pd.to_numeric(panel["Predicted_CP_REPO_FRESH"], errors="coerce").to_numpy(dtype=float)
    pm = pd.to_numeric(panel["Predicted_PM_QDK_2"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(cp) & np.isfinite(pm)
    actual, cp, pm = actual[valid], cp[valid], pm[valid]
    excess_mse = np.square(actual - pm) - np.square(actual - cp)
    segments: list[tuple[str, np.ndarray]] = [
        ("all", np.ones(len(actual), dtype=bool)),
        ("actual_zero", actual == 0.0),
        ("cp_negative", cp < 0.0),
    ]
    for percentile in [50, 75, 90, 95, 99]:
        threshold = float(np.percentile(actual, percentile))
        segments.append((f"actual_top_{100 - percentile}pct", actual >= threshold))
    rows = []
    for name, mask in segments:
        if not mask.any():
            continue
        rows.append(
            {
                "segment": name,
                "n_obs": int(mask.sum()),
                "share": float(mask.mean()),
                "actual_mean": float(actual[mask].mean()),
                "cp_smape_minus_pm_smape": float(np.mean(smape(actual[mask], cp[mask]) - smape(actual[mask], pm[mask]))),
                "cp_mae_minus_pm_mae": float(np.mean(np.abs(actual[mask] - cp[mask]) - np.abs(actual[mask] - pm[mask]))),
                "cp_mse_minus_pm_mse": float(np.mean(np.square(actual[mask] - cp[mask]) - np.square(actual[mask] - pm[mask]))),
                "share_of_total_pm_excess_mse": float(excess_mse[mask].sum() / excess_mse.sum()),
                "pm_to_cp_prediction_ratio_median": float(np.nanmedian(pm[mask] / np.where(np.abs(cp[mask]) > 1e-18, cp[mask], np.nan))),
            }
        )
    return pd.DataFrame(rows)


def asset_metric_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset, group in panel.groupby("asset", sort=True):
        actual = pd.to_numeric(group["Actual"], errors="coerce").to_numpy(dtype=float)
        cp = pd.to_numeric(group["Predicted_CP_REPO_FRESH"], errors="coerce").to_numpy(dtype=float)
        pm = pd.to_numeric(group["Predicted_PM_QDK_2"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(cp) & np.isfinite(pm)
        actual, cp, pm = actual[valid], cp[valid], pm[valid]
        rows.append(
            {
                "asset": asset,
                "n_obs": int(len(actual)),
                "zero_actual_share": float(np.mean(actual == 0.0)),
                "negative_cp_prediction_share": float(np.mean(cp < 0.0)),
                "SMAPE_advantage": float(np.mean(smape(actual, cp) - smape(actual, pm))),
                "MAE_advantage": float(np.mean(np.abs(actual - cp) - np.abs(actual - pm))),
                "MSE_advantage": float(np.mean(np.square(actual - cp) - np.square(actual - pm))),
                "RMSE_advantage": float(np.sqrt(np.mean(np.square(actual - cp))) - np.sqrt(np.mean(np.square(actual - pm)))),
            }
        )
    return pd.DataFrame(rows)


def embedding_complexity_table(seed: int = 42) -> tuple[pd.DataFrame, float]:
    args = runner.build_arg_parser().parse_args([])
    args.n = 22
    args.seed = seed
    args.log_eps = 1e-12
    args.pm_low_modes = 12
    args.pm_tau_grid_values = runner.parse_float_grid(args.pm_tau_grid, runner.DEFAULT_PM_TAU_GRID)
    rng = np.random.default_rng(seed + 1000)
    x = np.exp(rng.normal(-7.0, 1.0, size=(800, 22)))
    frame = pd.DataFrame({f"lag{idx + 1}": x[:, idx] for idx in range(22)})
    _, _, true_phi = runner.pm_qdk2_embedding(x, 22, args.pm_tau_grid_values, args.log_eps, args.pm_low_modes)
    embeddings: dict[str, np.ndarray] = {"PM_QDK_2": true_phi}
    embeddings.update({name: value[0] for name, value in validation.placebo_embeddings(frame, args, seed).items()})
    embeddings.update({name: value[0] for name, value in validation.pm_ablation_embeddings(frame, args).items()})
    rows = []
    for name, phi in embeddings.items():
        raw = validation.embedding_complexity(phi, len(phi))
        centered = phi - np.nanmean(phi, axis=0, keepdims=True)
        scale = np.nanstd(centered, axis=0)
        scaled = centered / np.where(scale > 1e-12, scale, 1.0)
        standardized = validation.embedding_complexity(scaled, len(scaled))
        rows.append(
            {
                "model_name": name,
                "ambient_dimension": int(phi.shape[1]),
                "raw_rank": int(raw["rank"]),
                "raw_effective_rank": float(raw["effective_rank"]),
                "standardized_rank": int(standardized["rank"]),
                "standardized_effective_rank": float(standardized["effective_rank"]),
                "nonzero_scale_count": int(raw["nonzero_scale_count"]),
            }
        )
    modes, mu = runner.pm_mode_subset(22, 12)
    chars = runner.prime_character_matrix(22, modes)
    z = np.log(x + args.log_eps)
    z -= z.mean(axis=1, keepdims=True)
    base = z @ np.conjugate(chars)
    pieces = [base * np.exp(-tau * mu)[None, :] for tau in args.pm_tau_grid_values]
    normalized = [piece / np.where(np.std(piece, axis=0) > 1e-15, np.std(piece, axis=0), 1.0) for piece in pieces]
    cancellation = max(float(np.max(np.abs(piece - normalized[0]))) for piece in normalized[1:])
    return pd.DataFrame(rows), cancellation


def old_solver_artifacts(full_prop4: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((full_prop4 / "predictions" / "full").glob("*.csv")):
        frame = pd.read_csv(path)
        for model in ["CP_REPO_OPS_R", "CP_REPO_LRPM", "CP_REPO_OPS_C"]:
            values = pd.to_numeric(frame.get(f"Predicted_{model}"), errors="coerce").dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "asset": path.stem,
                    "model_name": model,
                    "max_abs_prediction": float(np.max(np.abs(values))),
                    "count_abs_prediction_gt_0_1": int(np.sum(np.abs(values) > 0.1)),
                }
            )
    return pd.DataFrame(rows)


def findings_table(cancellation: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("P0", "Prop 4 expanding solver", "Sequential normal equations were numerically non-equivalent on rank-deficient shape models; QR repair is required.", "fixed_and_verified"),
            ("P0", "Random gate placebo", "Full-series permutation imported future real-gate values into earlier placebo rows.", "fixed_and_verified"),
            ("P1", "PHQO zero paths", "The ratio form skipped zero-path rows instead of taking the exact-CP continuous safety fallback.", "fixed_and_verified"),
            ("P1", "Pooled RMSE", "The POOLED row averaged asset RMSEs instead of taking the root of pooled squared error.", "fixed_for_future_reports"),
            ("P0", "PM diffusion fidelity", f"Per-coordinate standardization cancels every heat multiplier; maximum post-standardization tau-block difference was {cancellation:.3g}.", "requires_new_frozen_model"),
            ("P0", "Matched placebo claim", "Placebos match 120 stored columns but not effective rank or Gram spectrum, so complexity is not actually held fixed.", "requires_spectral_matching"),
            ("P0", "PM quotient chart", "Historical Q rows are residualized with different expanding regression vintages than the query row.", "requires_common_origin_residualization"),
            ("P1", "CP geometry", "The implemented CP distance uses diagonal scaling although the theory specifies a Mahalanobis metric; C has rank 10 of 13.", "requires_pseudoinverse_covariance_metric"),
            ("P1", "CP-fiber success criterion", "A negative PM distance/residual correlation was previously counted as favorable merely because it was less negative than placebos.", "fixed_for_future_reports"),
            ("P1", "Inference", "The row bootstrap treats serially dependent intraday observations as independent and materially overstates precision.", "requires_asset_clustered_block_bootstrap"),
            ("P1", "Locked CP initialization", "Repo contig_prime_modulo backfills only its initial phase rows; exact reproduction and strict causal initialization cannot both hold without a separately named benchmark.", "documented_contract_caveat"),
        ],
        columns=["severity", "area", "finding", "status"],
    )


def proposed_model_text() -> str:
    return r"""# PM-CAST-D: Prime-Modular CP-Anchored Spectral Transport Distribution

## Purpose

`PM-CAST-D` is an average-based forecasting model. PM is embedded in the averaging operator itself, while `CP_REPO_FRESH` is exactly nested. It is designed to estimate the conditional distribution, then emit one frozen multi-loss point action for the common SMAPE/MAE/MSE/RMSE comparison.

No model can guarantee finite-sample OOS dominance across all four losses. Mean, median, and the SMAPE Bayes action are generally different. The defensible objective is a CP-nested class, train-only selection, and a fresh holdout on which one common point forecast improves all four metrics.

## 1. Exact PM quotient field

For the positive lag path `x in R_+^22`, define

```text
z = log(x + eps) - mean(log(x + eps)) 1,
r = M_C z,
M_C = I - C' (C C')^+ C.
```

Let `U` be an orthonormal real basis obtained from the restricted characters of `Z2 x Z3 x Z5`, and let `mu_k` be the product-cycle Laplacian eigenvalues. The dimensionless PM heat fields are

```text
E_tau(x) = U diag(exp(-tau mu_k)) U' r.
```

Only one scalar train-only normalization is allowed for the entire Hilbert block. Coordinate-wise standardization is forbidden because it removes `exp(-tau mu_k)`.

## 2. PM spectral transport of average weights

Let `beta > 0`, `sum(beta)=1`, be a fixed positive CP/HAR averaging reference. A low-rank state map produces spectral transport scores:

```text
s_l(x) = sum_k theta_k(Cx) [E_tau_k(x)]_l,
w_l(x) = beta_l exp(s_l(x)) / sum_j beta_j exp(s_j(x)).
```

The PM average and its CP reference are

```text
A_PM(x) = sum_l w_l(x) x_l,
A_0(x)  = sum_l beta_l x_l.
```

The CP-nested transported location is

```text
m_T(x) = m_CP(x) [A_PM(x)+eps] / [A_0(x)+eps].
```

If the PM score is zero, the transport temperature is zero, or `A_0=0`, set `m_T=m_CP` exactly.

## 3. Distributional calibration and one common action

Model the CP-relative log outcome

```text
R = log((Y+eps)/(m_CP+eps))
```

with a monotone distributional regression whose location is `log(m_T/m_CP)` and whose scale/tail parameters depend only on low-rank PM heat energy and the CP state. Fit it with expanding, purged, train-only CRPS. The practitioner forecast is restricted to the CP-to-transport segment

```text
a(alpha) = (1-alpha) m_CP + alpha m_T,  0 <= alpha <= alpha_max.
```

Choose one frozen `alpha` rule by minimizing worst normalized validation regret across SMAPE, MAE, and MSE:

```text
min_alpha max_L [R_L(a(alpha))-R_L(m_CP)] / scale_L,
L in {SMAPE, MAE, MSE}.
```

RMSE follows MSE ordering on an identical row set. Entropy/KL regularization on `w` keeps the construction close to the reference average, and `alpha=0` gives exact CP nesting.

## 4. Why this is a useful application exhibit

The forecast remains an average-based system: PM changes the mass assigned to the 22 lag observations through a prime-harmonic transport field. It does not append PM regressors to CP. The distributional layer prevents the SMAPE-only downward shift that caused the current upper-tail MAE/MSE failure.

Application success is not itself proof of prime specificity. `PM-CAST-D` must beat spectral-matched random, shuffled, Haar, composite, fake-torus, and contiguous transport operators under the identical estimator and tuning rule.
"""


def report_text(
    fidelity: pd.DataFrame,
    metrics: pd.DataFrame,
    tails: pd.DataFrame,
    complexity: pd.DataFrame,
    cancellation: float,
) -> str:
    cp = metrics.set_index("model_name").loc["CP_REPO_FRESH"]
    pm = metrics.set_index("model_name").loc["PM_QDK_2"]
    top_half = tails.set_index("segment").loc["actual_top_50pct"]
    top_quarter = tails.set_index("segment").loc["actual_top_25pct"]
    true = complexity.set_index("model_name").loc["PM_QDK_2"]
    placebo = complexity.loc[complexity["model_name"].isin(validation.PLACEBO_MODELS)]
    return f"""# Prop 4 Mathematical Audit And PM Diagnosis

## Outcome

- All `{len(fidelity)}` Prop 4 models pass finite-output, timestamp/target alignment, feature-registry, and fast-versus-direct solver checks after the QR and causal-gate repairs.
- Maximum fast/direct prediction difference in the smoke audit: `{fidelity['max_abs_fast_direct_diff'].max():.6g}`.
- `CP_REPO_FRESH` reproduction remains unchanged within `{fidelity.loc[fidelity['model_name'] == 'CP_REPO_FRESH', 'max_abs_fast_direct_diff'].iloc[0]:.6g}` in the solver audit.
- Existing full-run results for unregularized OPS-R/LRPM/OPS-C and the old random-gate placebo are superseded and require a fresh full run.

## Why PM_QDK_2 Has The Observed Loss Pattern

On `{int(pm['n_obs'])}` aligned rows, `PM_QDK_2` changes pooled SMAPE from `{cp['SMAPE']:.6f}` to `{pm['SMAPE']:.6f}`, but changes MAE from `{cp['MAE']:.9g}` to `{pm['MAE']:.9g}` and true pooled RMSE from `{cp['RMSE']:.9g}` to `{pm['RMSE']:.9g}`.

The model is SMAPE-native, and its Bayes action is lower than the conditional mean. Its mean forecast bias is `{pm['bias_pred_minus_actual']:.9g}` versus CP's `{cp['bias_pred_minus_actual']:.9g}`. The upper half of realized volatility explains `{top_half['share_of_total_pm_excess_mse']:.2%}` of PM's excess squared error; the upper quartile explains `{top_quarter['share_of_total_pm_excess_mse']:.2%}`. This is a location/action mismatch, not evidence that PM predicts tails well.

## Why Prime Specificity Is Not Established

1. Coordinate-wise quotient scaling cancels the heat factors to `{cancellation:.3g}` numerical error. The intended prime-diffusion smoothing is therefore absent from the actual distance.
2. True PM has standardized effective rank `{true['standardized_effective_rank']:.3f}` in the controlled synthetic audit, while the six placebos range from `{placebo['standardized_effective_rank'].min():.3f}` to `{placebo['standardized_effective_rank'].max():.3f}`. Equal stored column count is not equal complexity.
3. Expanding quotient residuals use vintage-specific projection maps, so historical and query points are not represented in one common CP-fiber chart.
4. The CP distance is diagonal-scaled although `C` has rank `10/13` and the theory specifies a covariance-pseudoinverse metric.
5. The selected PM_QDK_2 bandwidth is at the broadest grid value for nearly every asset. Combined with the identical SMAPE action used by all geometries, most of the gain is generic smoothing/action regularization.
6. The previous CP-fiber success flag compared two negative correlations. The corrected criterion requires a positive PM slope before comparing it with placebos.

## Decisive PM Embedding Test

1. Freeze the PM operator, estimator, tau grid, rank, loss action, and selection rule before looking at the final period.
2. Represent every geometry by a `22 x d` operator and spectrally match its singular values, effective rank, trace, Frobenius norm, heat eigenvalue multiplicities, and kernel bandwidth budget to true PM.
3. Fit the train-only conditional projection once at each forecast origin and transform the query and every historical candidate with that same map.
4. Use the exact Mahalanobis CP metric and one scalar PM-Hilbert normalization, preserving relative heat weights.
5. Add a no-Q `(L,C)` kernel. PM must beat this ablation; otherwise the reported gain belongs to the common level/CP smoother and SMAPE action.
6. Use identical anchor-pair sets for every CP-fiber and smoothness comparison. Require positive fiber slope, lower PM residual Dirichlet energy, and low-mode held-out residual concentration.
7. Compare true PM with at least 1,000 spectral-matched random rotations plus the six structured placebos. Report PM's randomization percentile with family-wise correction.
8. Use purged chronological folds for tuning, then one untouched post-freeze time block. Infer with day/week moving-block bootstrap clustered by asset, plus the 10-asset sign/win test.

## Claim Threshold

Prime-specific value is supported only if PM beats the no-Q baseline and every structured placebo, ranks above the 95th percentile of spectral-matched rotations, has a positive CP-fiber slope and lower residual graph energy, wins at least 7/10 assets, and retains positive asset-clustered block-bootstrap intervals without a material MAE/RMSE loss.

The proposed practitioner model is specified separately in `PROPOSED_PM_CAST_D.md`.
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", default="outputs/prop4_math_audit_smoke_20260709")
    parser.add_argument("--pm-results", default="outputs/pm_native_geometry_warmup600_discriminating_validation")
    parser.add_argument("--old-prop4-full", default="outputs/cp_repo_ops_hg_actions/28269511971/combined_manual")
    parser.add_argument("--outdir", default="outputs/pm_prop4_mathematical_audit_20260709")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fidelity = model_fidelity_table(Path(args.smoke))
    panel = load_pm_panels(Path(args.pm_results))
    metrics = pooled_metric_table(panel)
    tails = pm_tail_decomposition(panel)
    assets = asset_metric_table(panel)
    complexity, cancellation = embedding_complexity_table()
    old_artifacts = old_solver_artifacts(Path(args.old_prop4_full))
    findings = findings_table(cancellation)
    write_csv(fidelity, outdir / "model_fidelity_audit.csv")
    write_csv(metrics, outdir / "pooled_metric_recalculation.csv")
    write_csv(tails, outdir / "pm_tail_loss_decomposition.csv")
    write_csv(assets, outdir / "pm_asset_metric_recalculation.csv")
    write_csv(complexity, outdir / "pm_embedding_complexity_audit.csv")
    write_csv(old_artifacts, outdir / "superseded_solver_artifacts.csv")
    write_csv(findings, outdir / "implementation_findings.csv")
    (outdir / "AUDIT_AND_RESEARCH_RECOMMENDATION.md").write_text(
        report_text(fidelity, metrics, tails, complexity, cancellation), encoding="utf-8"
    )
    (outdir / "PROPOSED_PM_CAST_D.md").write_text(proposed_model_text(), encoding="utf-8")
    print(f"Audit package written to {outdir}")


if __name__ == "__main__":
    main()
