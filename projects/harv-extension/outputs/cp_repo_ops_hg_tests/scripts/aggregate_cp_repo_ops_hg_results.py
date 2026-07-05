"""Lightweight aggregation for CP-REPO-OPS-HG per-asset artifacts.

This script intentionally avoids recomputing row-level model metrics from the
large full prediction panels. It combines per-asset result CSVs, rebuilds pooled
summary tables from those per-asset summaries, and writes validation logs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ASSETS = ["AAPL", "AMZN", "EEM", "FXI", "GLD", "GOOGL", "HYG", "QQQ", "SPY", "TLT"]
DEFAULT_MODELS = [
    "HAR_RV",
    "CP_REPO_FRESH",
    "RAW_PM",
    "RAW_CP_REPO_PLUS_PM",
    "CP_REPO_OPS_R",
    "CP_REPO_RIDGE_OPS_R",
    "CP_REPO_LRPM",
    "CP_REPO_RECENT_SLOPE",
    "CP_REPO_HAAR_SHAPE",
    "CP_REPO_OPS_C",
    "CP_REPO_RIDGE_OPS_C",
    "CP_REPO_GATED_RIDGE_OPS_C",
    "CP_REPO_OPS_HG",
    "RANDOM_RESIDUES_PLACEBO_REPO",
    "SHUFFLED_LAG_PM_PLACEBO_REPO",
    "RANDOM_GATE_PLACEBO_REPO",
    "RIDGE_AR22",
    "CP_REPO_OPS_K",
]
PLACEBO_MODELS = {
    "RANDOM_RESIDUES_PLACEBO_REPO",
    "SHUFFLED_LAG_PM_PLACEBO_REPO",
    "RANDOM_GATE_PLACEBO_REPO",
    "RIDGE_AR22",
}
DIAGNOSTIC_NON_PAPER_MODELS = {"CP_REPO_LRPM"}


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def parse_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [part.strip().upper() for part in str(value).split(",") if part.strip()]


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def find_asset_root(artifacts_dir: Path, asset: str) -> Path | None:
    candidates = [
        path
        for path in artifacts_dir.rglob(f"{asset}.csv")
        if path.as_posix().endswith(f"predictions/full/{asset}.csv")
    ]
    if not candidates:
        return None
    return candidates[0].parents[2]


def concat_result(asset_roots: dict[str, Path], filename: str) -> pd.DataFrame:
    frames = []
    for asset, root in asset_roots.items():
        path = root / "results" / filename
        if path.exists():
            frame = safe_read_csv(path)
            if "asset" not in frame.columns:
                frame["asset"] = asset
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def summarize_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    metric_cols = [
        "model_mean_SMAPE",
        "CP_mean_SMAPE",
        "mean_advantage_vs_CP",
        "median_advantage_vs_CP",
        "win_rate_vs_CP",
        "winsorized_mean_advantage_1_99",
        "trimmed_mean_advantage_10pct",
        "mean_gain_when_model_wins",
        "mean_loss_when_model_loses",
        "median_abs_advantage_vs_CP",
        "mean_abs_error_model",
        "mean_abs_error_CP",
        "mean_abs_error_advantage_vs_CP",
    ]
    rows = []
    group_cols = ["model_name", "condition"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        model_name, condition = keys
        weights = pd.to_numeric(group.get("n_obs", 0), errors="coerce").fillna(0)
        rec = {
            "model_name": model_name,
            "benchmark_model": "CP_REPO_FRESH",
            "asset": "POOLED",
            "condition": condition,
            "condition_label_type": (
                "not_conditioned" if condition == "all_observations" else "ex_post_descriptive"
            ),
            "n_obs": int(weights.sum()),
            "run_type": "full",
            "model_status": "pooled",
            "source": "pooled_from_asset_summaries",
            "benchmark_source": "pooled_from_asset_summaries",
            "source_path": "",
            "seed": np.nan,
            "phase": "",
            "is_paper_eligible": bool(
                group.get("is_paper_eligible", pd.Series([True] * len(group))).astype(bool).all()
                and model_name not in DIAGNOSTIC_NON_PAPER_MODELS
            ),
            "alignment_exact": bool(group.get("alignment_exact", pd.Series([True] * len(group))).astype(bool).all()),
            "actual_match": bool(group.get("actual_match", pd.Series([True] * len(group))).astype(bool).all()),
            "number_of_assets": int(group["asset"].nunique()) if "asset" in group else int(len(group)),
        }
        for col in metric_cols:
            if col in group.columns:
                rec[col] = weighted_mean(group[col], weights)
        if "mean_advantage_vs_CP" in group.columns:
            adv = pd.to_numeric(group["mean_advantage_vs_CP"], errors="coerce")
            rec["number_of_assets_positive"] = int((adv > 0).sum())
            rec["asset_sign_count"] = int((adv > 0).sum() - (adv < 0).sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_alternative_loss(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    group_cols = ["model_name", "condition", "condition_label_type", "loss_variant", "loss_metric"]
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, keys))
        weights = pd.to_numeric(group.get("n_obs", 0), errors="coerce").fillna(0)
        cp_loss = weighted_mean(group["CP_loss"], weights) if "CP_loss" in group else np.nan
        model_loss = weighted_mean(group["model_loss"], weights) if "model_loss" in group else np.nan
        adv = pd.to_numeric(group.get("advantage_CP_minus_model", np.nan), errors="coerce")
        rec.update(
            {
                "n_obs_total": int(weights.sum()),
                "n_assets": int(group["asset"].nunique()) if "asset" in group else int(len(group)),
                "pooled_CP_loss": cp_loss,
                "pooled_model_loss": model_loss,
                "pooled_advantage_CP_minus_model": cp_loss - model_loss if np.isfinite(cp_loss) and np.isfinite(model_loss) else np.nan,
                "equal_weight_asset_advantage": float(adv.mean()) if adv.notna().any() else np.nan,
                "assets_positive": int((adv > 0).sum()),
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def bool_column(frame: pd.DataFrame, name: str, default: bool = True) -> pd.Series:
    if name in frame.columns:
        return frame[name].astype(bool)
    return pd.Series([default] * len(frame), index=frame.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CP-REPO-OPS-HG per-asset artifacts")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--assets", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--copy-predictions", action="store_true")
    parser.add_argument("--run-audits", default="true")
    args = parser.parse_args()

    assets = parse_csv(args.assets, DEFAULT_ASSETS)
    models = parse_csv(args.models, DEFAULT_MODELS)
    run_audits = str(args.run_audits).strip().lower() in {"1", "true", "yes", "y"}
    artifacts_dir = Path(args.artifacts_dir)
    outdir = Path(args.outdir)
    result_dir = outdir / "results"
    paper_dir = outdir / "paper_tables"
    log_dir = outdir / "logs"
    for directory in [result_dir, paper_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    failures = []
    asset_roots = {}
    for asset in assets:
        root = find_asset_root(artifacts_dir, asset)
        if root is None:
            failures.append(f"{asset}: missing predictions/full/{asset}.csv")
            continue
        asset_roots[asset] = root
        if args.copy_predictions:
            pred_src = root / "predictions" / "full" / f"{asset}.csv"
            pred_dest = outdir / "predictions" / "full" / f"{asset}.csv"
            pred_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pred_src, pred_dest)
        for log_path in sorted((root / "logs").glob("*")) if (root / "logs").exists() else []:
            if log_path.is_file():
                dest = log_dir / asset
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(log_path, dest / log_path.name)

    registry_source = next((root / "model_registry.json" for root in asset_roots.values() if (root / "model_registry.json").exists()), None)
    if registry_source:
        shutil.copy2(registry_source, outdir / "model_registry.json")

    result_files = [
        "run_metadata.csv",
        "run_failures.csv",
        "model_feature_manifest.csv",
        "placebo_diagnostics.csv",
        "no_lookahead_audit.csv",
        "repo_cp_reproduction_audit.csv",
        "fast_slow_equivalence_audit.csv",
        "strong_fast_slow_equivalence_audit.csv",
        "raw_cp_pm_reproduction_audit.csv",
        "alignment_audit.csv",
        "model_overall_by_asset.csv",
        "model_conditional_by_asset.csv",
        "inference_summary.csv",
        "alternative_loss_summary_by_asset.csv",
    ]
    combined = {name: concat_result(asset_roots, name) for name in result_files}
    for name, frame in combined.items():
        write_csv(frame, result_dir / name)

    overall_by_asset = combined["model_overall_by_asset.csv"]
    conditional_by_asset = combined["model_conditional_by_asset.csv"]
    pooled = summarize_metric_rows(conditional_by_asset)
    if not pooled.empty:
        pooled["aggregation_method"] = "asset_summary_weighted"
    ablation = pooled[pooled["condition"] == "all_observations"].copy() if not pooled.empty else pd.DataFrame()
    if not ablation.empty:
        order = {name: idx for idx, name in enumerate(DEFAULT_MODELS)}
        ablation["ladder_order"] = ablation["model_name"].map(lambda name: order.get(str(name), 999))
        ablation = ablation.sort_values(["ladder_order", "model_name"])
    placebo = ablation[ablation["model_name"].isin(PLACEBO_MODELS)].copy() if not ablation.empty else pd.DataFrame()
    alt_loss = summarize_alternative_loss(combined["alternative_loss_summary_by_asset.csv"])
    if not alt_loss.empty:
        alt_loss["aggregation_method"] = "asset_summary_weighted"

    write_csv(pooled, result_dir / "model_conditional_pooled.csv")
    write_csv(ablation, result_dir / "ablation_ladder_summary.csv")
    write_csv(placebo, result_dir / "placebo_summary.csv")
    write_csv(alt_loss, result_dir / "alternative_loss_summary.csv")
    write_csv(overall_by_asset, result_dir / "model_overall_by_asset.csv")
    write_csv(conditional_by_asset, result_dir / "model_conditional_by_asset.csv")

    paper_main = ablation[bool_column(ablation, "is_paper_eligible") & (~ablation["model_name"].isin(DIAGNOSTIC_NON_PAPER_MODELS))].copy() if not ablation.empty else pd.DataFrame()
    paper_cond = pooled[bool_column(pooled, "is_paper_eligible") & (~pooled["model_name"].isin(DIAGNOSTIC_NON_PAPER_MODELS))].copy() if not pooled.empty else pd.DataFrame()
    paper_placebo = placebo[bool_column(placebo, "is_paper_eligible")].copy() if not placebo.empty else pd.DataFrame()
    write_csv(paper_main, paper_dir / "main_model_summary.csv")
    write_csv(paper_cond, paper_dir / "conditional_summary.csv")
    write_csv(paper_placebo, paper_dir / "placebo_summary.csv")

    metadata = combined["run_metadata.csv"]
    run_failures = combined["run_failures.csv"]
    repo_audit = combined["repo_cp_reproduction_audit.csv"]
    fast_audit = combined["fast_slow_equivalence_audit.csv"]
    strong_audit = combined["strong_fast_slow_equivalence_audit.csv"]
    lookahead = combined["no_lookahead_audit.csv"]
    manifest = combined["model_feature_manifest.csv"]

    repo_pass = True if not run_audits else (not repo_audit.empty and repo_audit.get("passes", pd.Series(dtype=bool)).astype(bool).all())
    fast_pass = True if not run_audits else (not fast_audit.empty and fast_audit.get("passes", pd.Series(dtype=bool)).astype(bool).all())
    strong_pass = True
    if run_audits and not strong_audit.empty:
        strong_pass = bool(strong_audit.get("passes", pd.Series(dtype=bool)).astype(bool).all())

    audit_rows = [
        {"check": "assets_present", "passed": set(asset_roots) == set(assets), "detail": ",".join(sorted(asset_roots))},
        {"check": "metadata_complete_rows", "passed": len(metadata) >= len(asset_roots) * min(len(models), len(DEFAULT_MODELS)) and (metadata.get("status", pd.Series(dtype=str)).astype(str) == "complete").all(), "detail": f"{len(metadata)} metadata rows"},
        {"check": "run_failures_empty", "passed": run_failures.empty, "detail": f"{len(run_failures)} failure rows"},
        {"check": "repo_cp_reproduction_pass", "passed": repo_pass, "detail": "required" if run_audits else "skipped because run_audits=false"},
        {"check": "fast_slow_pass", "passed": fast_pass, "detail": "required" if run_audits else "skipped because run_audits=false"},
        {"check": "strong_fast_slow_pass", "passed": strong_pass, "detail": "required if present"},
        {"check": "no_lookahead_pass", "passed": (not lookahead.empty and lookahead.get("passed", pd.Series(dtype=bool)).astype(bool).all()), "detail": ""},
        {"check": "global_level_sum_pass", "passed": (not manifest.empty and manifest.get("global_level_sum_pass", pd.Series(dtype=bool)).astype(bool).all()), "detail": ""},
    ]
    if failures:
        audit_rows.append({"check": "artifact_discovery", "passed": False, "detail": "; ".join(failures)})
    audit = pd.DataFrame(audit_rows)
    write_csv(audit, result_dir / "aggregate_audit_summary.csv")
    (log_dir / "aggregate_validation.txt").write_text(audit.to_string(index=False) + "\n", encoding="utf-8")

    readme = f"""# CP-REPO-OPS-HG Combined Artifact

Generated: {utc_now()}

This combined artifact was built from per-asset GitHub Actions artifacts using
lightweight summary aggregation. The workflow runs `tables-only` afterward when
prediction panels are available so final paper tables can be regenerated from
row-level saved predictions without refitting models.

- Assets found: {','.join(sorted(asset_roots))}
- Missing assets: {','.join(sorted(set(assets) - set(asset_roots))) or '[none]'}
- Models requested: {','.join(models)}
- Diagnostic non-paper model excluded from paper tables: CP_REPO_LRPM

See `results/aggregate_audit_summary.csv` and `logs/aggregate_validation.txt`.
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")
    if bool(audit["passed"].astype(bool).all()) and not paper_main.empty:
        (outdir / "SUCCESS.txt").write_text("Combined CP-REPO-OPS-HG artifact completed successfully\n", encoding="utf-8")
    else:
        (outdir / "AGGREGATE_NEEDS_REVIEW.txt").write_text("Combined artifact was written but one or more aggregate checks failed\n", encoding="utf-8")

    print(json.dumps({"outdir": str(outdir), "assets_found": sorted(asset_roots), "checks_passed": bool(audit["passed"].astype(bool).all())}, indent=2))
    if not bool(audit["passed"].astype(bool).all()):
        sys.exit(1)


if __name__ == "__main__":
    main()
