"""
Build the internal PM/CP research package from completed source runs.

This script is intentionally tables-first. It consolidates the older CP+PM
incremental evidence, the corrected repo-CP 18-model ladder, and the selective
overlay audit outputs without copying large prediction panels into the package.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTDIR = PROJECT_ROOT / "outputs" / "pm_research_internal_package"
DEFAULT_PROP3_DIR = PROJECT_ROOT / "outputs" / "cp_pm_incremental_tests"
DEFAULT_PROP4_RUN = PROJECT_ROOT / "outputs" / "cp_repo_ops_hg_actions" / "28269511971"
DEFAULT_PRED_ROOT = DEFAULT_PROP4_RUN / "combined_manual" / "predictions" / "full"
DEFAULT_OVERLAY_DIR = PROJECT_ROOT / "outputs" / "pm_selective_overlay_tests"
ASSETS = ["AAPL", "AMZN", "EEM", "FXI", "GLD", "GOOGL", "HYG", "QQQ", "SPY", "TLT"]
PROP4_RUN_MODELS = [
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
PLACEBO_MODELS = [
    "RANDOM_RESIDUES_PLACEBO_REPO",
    "SHUFFLED_LAG_PM_PLACEBO_REPO",
    "RANDOM_GATE_PLACEBO_REPO",
    "RIDGE_AR22",
]


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ensure_dirs(outdir: Path) -> None:
    for name in [
        "tables",
        "figures",
        "audits",
        "source_runs",
        "latex",
        "prop3_raw_pm_diagnostics/tables",
        "prop3_raw_pm_diagnostics/figures",
        "prop4_corrected_repo_cp_ladder/tables",
        "prop4_corrected_repo_cp_ladder/audits",
        "prop4_corrected_repo_cp_ladder/figures",
        "prop5_overlay_exploratory/tables",
        "prop5_overlay_exploratory/figures",
    ]:
        (outdir / name).mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def copy_if_exists(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def copy_text_strip_trailing(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    cleaned = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    dest.write_text(cleaned, encoding="utf-8")
    return True


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def make_prop4_model_status(overall: pd.DataFrame) -> pd.DataFrame:
    result_models = set(overall["model_name"].astype(str)) if not overall.empty and "model_name" in overall.columns else set()
    rows = []
    for model in PROP4_RUN_MODELS:
        rows.append(
            {
                "model_name": model,
                "status": "full_run_result" if model in result_models else "missing_from_prop4_table",
                "in_run_28269511971": model in result_models,
                "paper_result_table": model in result_models and model != "CP_REPO_LRPM",
                "limitation": "rank_deficient_unregularized_diagnostic" if model == "CP_REPO_LRPM" else "",
            }
        )
    rows.append(
        {
            "model_name": "CP_REPO_RIDGE_LRPM",
            "status": "implemented_after_run_not_in_prop4_table",
            "in_run_28269511971": False,
            "paper_result_table": False,
            "limitation": "implementation-ready in runner; requires new full run before entering result tables",
        }
    )
    return pd.DataFrame(rows)


def prediction_row_counts(pred_root: Path) -> pd.DataFrame:
    rows = []
    for asset in ASSETS:
        path = pred_root / f"{asset}.csv"
        if not path.exists():
            rows.append({"asset": asset, "prediction_file": str(path), "exists": False, "n_rows": 0, "n_models": 0})
            continue
        header = pd.read_csv(path, nrows=0)
        n_rows = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        n_models = len([col for col in header.columns if col.startswith("Predicted_")])
        rows.append(
            {
                "asset": asset,
                "prediction_file": str(path.relative_to(PROJECT_ROOT)),
                "exists": True,
                "n_rows": n_rows,
                "n_models": n_models,
            }
        )
    return pd.DataFrame(rows)


def source_manifest(prop3_dir: Path, prop4_run: Path, overlay_dir: Path, outdir: Path) -> pd.DataFrame:
    candidates = [
        ("prop3_raw_pm", prop3_dir / "README.md", "Older CP+PM incremental README"),
        ("prop3_raw_pm", prop3_dir / "results" / "cp_recompute_validation.csv", "Old CP recomputation validation"),
        (
            "prop3_raw_pm",
            prop3_dir / "results" / "fresh_conditional_hybrid_summary_pooled.csv",
            "Raw CP+PM conditional summary",
        ),
        ("prop3_raw_pm", prop3_dir / "results" / "top_fresh_hybrid_conditions.csv", "Top raw CP+PM conditions"),
        ("prop4_ladder", prop4_run / "audit_report.md", "Manual post-run audit report"),
        (
            "prop4_ladder",
            prop4_run / "audit_recomputed" / "recomputed_model_overall_summary.csv",
            "Corrected 18-model overall SMAPE summary",
        ),
        (
            "prop4_ladder",
            prop4_run / "audit_recomputed" / "recomputed_model_conditional_summary.csv",
            "Corrected 18-model conditional summary",
        ),
        (
            "prop4_ladder",
            prop4_run / "audit_recomputed" / "alternative_loss_summary.csv",
            "Alternative loss summary",
        ),
        (
            "prop4_ladder",
            prop4_run / "audit_recomputed" / "residual_perturbation_diagnostics.csv",
            "Residual perturbation diagnostics",
        ),
        (
            "audit",
            prop4_run / "combined_manual" / "results" / "repo_cp_reproduction_audit.csv",
            "10-asset repo CP reproduction audit",
        ),
        (
            "audit",
            prop4_run / "combined_manual" / "results" / "no_lookahead_audit.csv",
            "10-asset no-lookahead audit",
        ),
        ("prop5_overlay", prop4_run / "audit_recomputed" / "switching_overlay_summary.csv", "Selective overlay summary"),
        (
            "prop5_overlay",
            prop4_run / "audit_recomputed" / "switching_overlay_summary_by_asset.csv",
            "Selective overlay by asset",
        ),
        ("audit", prop4_run / "audit_recomputed" / "strong_fast_slow_equivalence_audit.csv", "Strong fast/slow audit"),
        ("audit", prop4_run / "audit_recomputed" / "feature_purity_audit.csv", "Feature purity audit"),
        ("audit", prop4_run / "audit_recomputed" / "ridge_penalty_audit.csv", "Ridge penalty audit"),
        ("audit", prop4_run / "audit_recomputed" / "audit_summary.json", "Audit summary JSON"),
        (
            "prop5_overlay_suite",
            overlay_dir / "results" / "overlay_pooled_summary.csv",
            "Reproducible overlay suite regenerated from saved prediction panels",
        ),
        (
            "prop5_overlay_suite",
            overlay_dir / "results" / "overlay_threshold_selection.csv",
            "Overlay threshold provenance and validation choices",
        ),
        (
            "prop5_overlay_suite",
            overlay_dir / "results" / "overlay_placebo_summary.csv",
            "Overlay random-gate/placebo comparison",
        ),
    ]
    rows = []
    for group, path, description in candidates:
        exists = path.exists()
        rows.append(
            {
                "group": group,
                "description": description,
                "source_path": str(path.relative_to(PROJECT_ROOT)) if path.exists() else str(path),
                "exists": exists,
            }
        )
    write_csv(pd.DataFrame(rows), outdir / "source_runs" / "source_run_manifest.csv")
    return pd.DataFrame(rows)


def make_tables(prop3_dir: Path, prop4_run: Path, overlay_dir: Path, outdir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    prop3_map = {
        "prop3_cp_recompute_validation.csv": prop3_dir / "results" / "cp_recompute_validation.csv",
        "prop3_raw_cp_pm_conditional_summary.csv": prop3_dir / "results" / "fresh_conditional_hybrid_summary_pooled.csv",
        "prop3_top_raw_cp_pm_conditions.csv": prop3_dir / "results" / "top_fresh_hybrid_conditions.csv",
    }
    for dest_name, src in prop3_map.items():
        copy_if_exists(src, outdir / "tables" / dest_name)
        tables[dest_name] = read_csv(src)

    audit = prop4_run / "audit_recomputed"
    prop4_map = {
        "prop4_overall_smape_summary.csv": audit / "recomputed_model_overall_summary.csv",
        "prop4_conditional_summary.csv": audit / "recomputed_model_conditional_summary.csv",
        "prop4_alternative_loss_summary.csv": audit / "alternative_loss_summary.csv",
        "prop4_alternative_loss_summary_by_asset.csv": audit / "alternative_loss_summary_by_asset.csv",
        "prop5_selective_overlay_summary.csv": audit / "switching_overlay_summary.csv",
        "prop5_selective_overlay_by_asset.csv": audit / "switching_overlay_summary_by_asset.csv",
        "prop4_residual_perturbation_diagnostics.csv": audit / "residual_perturbation_diagnostics.csv",
        "prop4_gate_condition_summary.csv": audit / "gate_condition_summary.csv",
    }
    for dest_name, src in prop4_map.items():
        copy_if_exists(src, outdir / "tables" / dest_name)
        tables[dest_name] = read_csv(src)

    overall = tables.get("prop4_overall_smape_summary.csv", pd.DataFrame())
    if not overall.empty:
        placebo = overall[overall["model_name"].astype(str).isin(PLACEBO_MODELS)].copy()
        write_csv(placebo, outdir / "tables" / "prop4_placebo_summary.csv")
        tables["prop4_placebo_summary.csv"] = placebo
        status = make_prop4_model_status(overall)
        write_csv(status, outdir / "tables" / "prop4_model_status.csv")
        tables["prop4_model_status.csv"] = status

    overlay_map = {
        "prop5_selective_overlay_recomputed_from_predictions.csv": overlay_dir / "results" / "overlay_pooled_summary.csv",
        "prop5_selective_overlay_threshold_selection.csv": overlay_dir / "results" / "overlay_threshold_selection.csv",
        "prop5_selective_overlay_placebo_summary.csv": overlay_dir / "results" / "overlay_placebo_summary.csv",
        "prop5_selective_overlay_loss_breakdown.csv": overlay_dir / "results" / "overlay_loss_breakdown.csv",
    }
    for dest_name, src in overlay_map.items():
        copy_if_exists(src, outdir / "tables" / dest_name)
        tables[dest_name] = read_csv(src)

    audit_map = {
        "cp_reproduction_audit.csv": prop4_run / "combined_manual" / "results" / "repo_cp_reproduction_audit.csv",
        "no_lookahead_audit.csv": prop4_run / "combined_manual" / "results" / "no_lookahead_audit.csv",
        "strong_fast_slow_equivalence_audit.csv": audit / "strong_fast_slow_equivalence_audit.csv",
        "feature_purity_audit.csv": audit / "feature_purity_audit.csv",
        "ridge_penalty_audit.csv": audit / "ridge_penalty_audit.csv",
        "manual_cp_reproduction_spotcheck.csv": audit / "manual_cp_reproduction_spotcheck.csv",
        "manual_overall_diffs_gt_1e-10.csv": audit / "manual_overall_diffs_gt_1e-10.csv",
        "manual_conditional_diffs_gt_1e-10.csv": audit / "manual_conditional_diffs_gt_1e-10.csv",
    }
    for dest_name, src in audit_map.items():
        copy_if_exists(src, outdir / "audits" / dest_name)

    copy_text_strip_trailing(prop4_run / "audit_report.md", outdir / "audits" / "source_audit_report.md")
    copy_text_strip_trailing(prop3_dir / "README.md", outdir / "source_runs" / "cp_pm_incremental_tests_README.md")
    return tables


def make_figures(tables: dict[str, pd.DataFrame], outdir: Path) -> None:
    prop3 = tables.get("prop3_raw_cp_pm_conditional_summary.csv", pd.DataFrame())
    if not prop3.empty and "equal_weight_asset_mean_advantage" in prop3.columns:
        plot = prop3.copy()
        fig, ax = plt.subplots(figsize=(11, 6))
        colors = np.where(plot["equal_weight_asset_mean_advantage"] >= 0, "#2F6F73", "#A64B4B")
        ax.barh(plot["condition"], plot["equal_weight_asset_mean_advantage"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Prop 3 raw CP+PM advantage by condition")
        ax.set_xlabel("Equal-weight SMAPE advantage vs CP")
        fig.tight_layout()
        fig.savefig(outdir / "figures" / "prop3_conditional_advantages.png", dpi=160)
        fig.savefig(outdir / "prop3_raw_pm_diagnostics" / "figures" / "prop3_condition_advantages.png", dpi=160)
        plt.close(fig)

    overall = tables.get("prop4_overall_smape_summary.csv", pd.DataFrame())
    if not overall.empty and "equal_weight_asset_mean_advantage_vs_CP" in overall.columns:
        plot = overall.copy()
        plot["order"] = plot["model_name"].map(lambda name: PROP4_RUN_MODELS.index(name) if name in PROP4_RUN_MODELS else 999)
        plot = plot.sort_values("equal_weight_asset_mean_advantage_vs_CP", ascending=True)
        fig, ax = plt.subplots(figsize=(11, 8))
        colors = np.where(plot["equal_weight_asset_mean_advantage_vs_CP"] >= 0, "#2F6F73", "#A64B4B")
        ax.barh(plot["model_name"], plot["equal_weight_asset_mean_advantage_vs_CP"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Always-on model SMAPE advantage vs repo CP")
        ax.set_xlabel("Equal-weight asset mean SMAPE advantage")
        fig.tight_layout()
        fig.savefig(outdir / "figures" / "prop4_overall_smape_advantage.png", dpi=160)
        fig.savefig(outdir / "figures" / "prop4_overall_smape_advantages.png", dpi=160)
        fig.savefig(outdir / "prop4_corrected_repo_cp_ladder" / "figures" / "prop4_overall_smape_bar.png", dpi=160)
        plt.close(fig)

    alt = tables.get("prop4_alternative_loss_summary.csv", pd.DataFrame())
    if not alt.empty:
        selected_models = [
            "CP_REPO_OPS_K",
            "CP_REPO_RIDGE_OPS_R",
            "CP_REPO_OPS_HG",
            "CP_REPO_GATED_RIDGE_OPS_C",
            "RAW_CP_REPO_PLUS_PM",
        ]
        base = alt[
            (alt["condition"].astype(str) == "all_rows")
            & (alt["loss_variant"].astype(str) == "standard")
            & (alt["model_name"].astype(str).isin(selected_models))
            & (alt["loss_metric"].astype(str).isin(["SMAPE", "MAE", "RMSE"]))
        ].copy()
        if not base.empty:
            pivot = base.pivot_table(
                index="model_name",
                columns="loss_metric",
                values="pooled_advantage_cp_minus_model",
                aggfunc="first",
            ).reindex(selected_models)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].barh(pivot.index, pivot["SMAPE"], color="#A64B4B")
            axes[0].axvline(0, color="black", linewidth=0.8)
            axes[0].set_title("SMAPE advantage")
            axes[0].set_xlabel("SMAPE points")
            scaled = pivot[["MAE", "RMSE"]].copy()
            scaled["MAE"] = scaled["MAE"] * 1_000_000.0
            scaled["RMSE"] = scaled["RMSE"] * 1_000_000.0
            y = np.arange(len(scaled.index))
            axes[1].barh(y - 0.18, scaled["MAE"], height=0.35, label="MAE x 1e6", color="#2F6F73")
            axes[1].barh(y + 0.18, scaled["RMSE"], height=0.35, label="RMSE x 1e6", color="#7A4E9E")
            axes[1].set_yticks(y)
            axes[1].set_yticklabels(scaled.index)
            axes[1].axvline(0, color="black", linewidth=0.8)
            axes[1].set_title("Non-SMAPE advantages")
            axes[1].legend()
            fig.suptitle("Prop 4 alternative losses")
            fig.tight_layout()
            fig.savefig(outdir / "figures" / "prop4_alternative_losses.png", dpi=160)
            fig.savefig(outdir / "prop4_corrected_repo_cp_ladder" / "figures" / "prop4_alt_loss_bar.png", dpi=160)
            plt.close(fig)

    conditional = tables.get("prop4_conditional_summary.csv", pd.DataFrame())
    if not conditional.empty and "equal_weight_asset_mean_advantage_vs_CP" in conditional.columns:
        models = [
            "RAW_CP_REPO_PLUS_PM",
            "CP_REPO_RIDGE_OPS_R",
            "CP_REPO_RECENT_SLOPE",
            "CP_REPO_HAAR_SHAPE",
            "CP_REPO_GATED_RIDGE_OPS_C",
            "CP_REPO_OPS_HG",
            "CP_REPO_OPS_K",
        ]
        conditions = [
            "all_observations",
            "cluster_entry_loose",
            "cluster_exit_loose",
            "recent_spike_position",
            "older_spike_position",
            "high_within_block_dispersion",
            "entry_with_recent_spike",
        ]
        matrix = conditional[
            conditional["model_name"].astype(str).isin(models)
            & conditional["condition"].astype(str).isin(conditions)
        ].pivot_table(
            index="model_name",
            columns="condition",
            values="equal_weight_asset_mean_advantage_vs_CP",
            aggfunc="first",
        ).reindex(index=models, columns=conditions)
        if not matrix.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            data = matrix.to_numpy(dtype=float)
            vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
            vmax = max(vmax, 0.1)
            im = ax.imshow(data, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
            ax.set_xticks(np.arange(len(matrix.columns)))
            ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
            ax.set_yticks(np.arange(len(matrix.index)))
            ax.set_yticklabels(matrix.index)
            ax.set_title("Prop 4 conditional SMAPE advantage heatmap")
            fig.colorbar(im, ax=ax, label="Equal-weight SMAPE advantage")
            fig.tight_layout()
            fig.savefig(outdir / "figures" / "prop4_conditional_heatmap.png", dpi=160)
            fig.savefig(outdir / "prop4_corrected_repo_cp_ladder" / "figures" / "prop4_conditional_heatmap.png", dpi=160)
            plt.close(fig)

    overlay = tables.get("prop5_selective_overlay_summary.csv", pd.DataFrame())
    if not overlay.empty and "mean_advantage_overall_equal_weight" in overlay.columns:
        plot = overlay.sort_values("mean_advantage_overall_equal_weight", ascending=False).head(12).copy()
        labels = plot["model_name"].astype(str) + "\n" + plot["strategy"].astype(str)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(labels.iloc[::-1], plot["mean_advantage_overall_equal_weight"].iloc[::-1], color="#7A4E9E")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Selective overlay SMAPE advantage")
        ax.set_xlabel("Equal-weight overall SMAPE advantage")
        fig.tight_layout()
        fig.savefig(outdir / "figures" / "prop5_selective_overlay_summary.png", dpi=160)
        plt.close(fig)


def first_value(frame: pd.DataFrame, column: str, default: str = "n/a") -> str:
    if frame.empty or column not in frame.columns:
        return default
    value = frame[column].iloc[0]
    if pd.isna(value):
        return default
    return str(value)


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "Unavailable."
    data = frame.copy()
    if max_rows is not None:
        data = data.head(max_rows)
    data = data.fillna("")
    columns = [str(col) for col in data.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in data.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in data.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def slim(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    keep = [col for col in columns if col in frame.columns]
    return frame[keep].copy()


def write_prop_subfolders(outdir: Path, tables: dict[str, pd.DataFrame]) -> None:
    prop3_dir = outdir / "prop3_raw_pm_diagnostics"
    prop4_dir = outdir / "prop4_corrected_repo_cp_ladder"
    prop5_dir = outdir / "prop5_overlay_exploratory"

    prop3 = tables.get("prop3_raw_cp_pm_conditional_summary.csv", pd.DataFrame())
    prop3_top = tables.get("prop3_top_raw_cp_pm_conditions.csv", pd.DataFrame())
    write_csv(prop3, prop3_dir / "tables" / "prop3_raw_cp_pm_conditional_summary.csv")
    write_csv(prop3_top, prop3_dir / "tables" / "prop3_top_conditions.csv")
    (prop3_dir / "README.md").write_text(
        "\n".join(
            [
                "# Prop 3 Raw PM Diagnostics",
                "",
                "Purpose: diagnose whether global/raw PM is useful as a standalone or always-on incremental signal.",
                "",
                "Conclusion: raw/global PM is noisy overall but has conditional signal in entry/recent-spike regimes.",
                "",
                "This folder is diagnostic background, not a main paper theorem.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (prop3_dir / "model_logic.md").write_text(
        "\n".join(
            [
                "# Prop 3 Model Logic",
                "",
                "Correct CP anchor: `CP_REPO_FRESH = RV/HAR + contig_prime_modulo(vol.copy(), n=22, per_day_normalize=False)`.",
                "",
                "Correct PM construction: `PM = add_prime_modulo_terms(..., n=22)` using the repo minimal-prime / CRT-style construction. For `n=22`, the relevant minimal primes are expected to include `2, 3, 5`.",
                "",
                "Raw CP+PM tests whether adding raw prime-residue averages to the CP/HAR design improves forecasts. It is deliberately not a cleaned OPS model.",
                "",
                "The stale `CPB_B*` four-block lag means are diagnostic only and are not the primary CP benchmark.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prop3_summary_cols = [
        "condition",
        "total_n_obs",
        "number_of_assets",
        "equal_weight_asset_mean_advantage",
        "equal_weight_asset_hybrid_win_rate",
        "number_of_assets_where_hybrid_beats_CP",
    ]
    (prop3_dir / "results_summary.md").write_text(
        "# Prop 3 Results Summary\n\n"
        "Known result pattern: all observations negative, cluster-entry positive, recent-spike positive, cluster-exit negative, older-spike negative.\n\n"
        + markdown_table(slim(prop3, prop3_summary_cols))
        + "\n\nInterpretation: raw/global PM is noisy overall but has conditional signal in entry/recent-spike regimes.\n",
        encoding="utf-8",
    )

    prop4_table_map = {
        "prop4_overall_smape_summary.csv": tables.get("prop4_overall_smape_summary.csv", pd.DataFrame()),
        "prop4_conditional_summary.csv": tables.get("prop4_conditional_summary.csv", pd.DataFrame()),
        "prop4_alternative_loss_summary.csv": tables.get("prop4_alternative_loss_summary.csv", pd.DataFrame()),
        "prop4_residual_perturbation_diagnostics.csv": tables.get("prop4_residual_perturbation_diagnostics.csv", pd.DataFrame()),
        "prop4_placebo_summary.csv": tables.get("prop4_placebo_summary.csv", pd.DataFrame()),
        "prop4_model_status.csv": tables.get("prop4_model_status.csv", pd.DataFrame()),
    }
    for name, frame in prop4_table_map.items():
        write_csv(frame, prop4_dir / "tables" / name)
    for name in [
        "cp_reproduction_audit.csv",
        "strong_fast_slow_equivalence_audit.csv",
        "feature_purity_audit.csv",
        "ridge_penalty_audit.csv",
    ]:
        copy_if_exists(outdir / "audits" / name, prop4_dir / "audits" / name)

    (prop4_dir / "README.md").write_text(
        "\n".join(
            [
                "# Prop 4 Corrected Repo-CP Ladder",
                "",
                "Purpose: test whether PM/path-order features add value inside average-based forecasting models.",
                "",
                "Status: not proven globally under headline SMAPE.",
                "",
                "Source run: GitHub Actions run `28269511971` from `Harman6139/money-making`, branch `cp-pm-incremental-actions`.",
                "",
                "Benchmark: `CP_REPO_FRESH`, using repo `contig_prime_modulo` with repo `RV*` and `CP_*` feature columns.",
                "",
                "`CP_REPO_RIDGE_LRPM` is implemented after the source run and is listed as `implemented_after_run_not_in_prop4_table`; it must not be read as part of run `28269511971` results.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (prop4_dir / "model_logic.md").write_text(
        "\n".join(
            [
                "# Prop 4 Model Logic",
                "",
                "Correct CP:",
                "",
                "`CP_REPO_FRESH = RV/HAR + contig_prime_modulo(vol.copy(), n=22, per_day_normalize=False)`",
                "",
                "Correct PM:",
                "",
                "`PM = add_prime_modulo_terms(..., n=22)` using the repo minimal-prime construction.",
                "",
                "Main model form:",
                "",
                "`y_{t+1} = alpha + beta'C_t + theta'Z_t + eps`",
                "",
                "Partial ridge form:",
                "",
                "`min ||y - alpha - C beta - Z theta||^2 + lambda_Z ||theta||^2`",
                "",
                "Gated/hybrid form:",
                "",
                "`y_{t+1} = alpha + beta'C_t + g_t(theta_C'Z_C,t + theta_R'Z_R,t) + eps`",
                "",
                "`CPB_B*` four-block lag means are diagnostic only and are forbidden as the primary CP benchmark.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    overall = tables.get("prop4_overall_smape_summary.csv", pd.DataFrame())
    overall_cols = [
        "model_name",
        "n_obs_total",
        "n_assets",
        "equal_weight_asset_mean_advantage_vs_CP",
        "equal_weight_asset_win_rate_vs_CP",
        "assets_positive",
    ]
    alt = tables.get("prop4_alternative_loss_summary.csv", pd.DataFrame())
    if not alt.empty:
        alt_show = alt[
            (alt["condition"].astype(str) == "all_rows")
            & (alt["loss_variant"].astype(str) == "standard")
            & (alt["loss_metric"].astype(str).isin(["SMAPE", "MAE", "MSE", "RMSE", "log_RV_MSE"]))
        ].copy()
    else:
        alt_show = pd.DataFrame()
    (prop4_dir / "results_summary.md").write_text(
        "# Prop 4 Results Summary\n\n"
        "Main conclusion: no always-on challenger beats `CP_REPO_FRESH` under headline SMAPE. Prop 4 is not proven as a broad always-on claim. Some models improve MAE/MSE/RMSE and some conditional regimes are positive, so PM/path-order information may still be useful in narrower settings.\n\n"
        "## Overall SMAPE\n\n"
        + markdown_table(slim(overall, overall_cols))
        + "\n\n## Alternative Losses\n\n"
        + markdown_table(
            slim(
                alt_show,
                [
                    "model_name",
                    "loss_metric",
                    "n_obs_total",
                    "pooled_advantage_cp_minus_model",
                    "assets_positive",
                ],
            ),
            max_rows=40,
        )
        + "\n",
        encoding="utf-8",
    )

    overlay_recomputed = tables.get("prop5_selective_overlay_recomputed_from_predictions.csv", pd.DataFrame())
    overlay_threshold = tables.get("prop5_selective_overlay_threshold_selection.csv", pd.DataFrame())
    overlay_placebo = tables.get("prop5_selective_overlay_placebo_summary.csv", pd.DataFrame())
    write_csv(overlay_recomputed, prop5_dir / "tables" / "prop5_overlay_recomputed_from_predictions.csv")
    write_csv(overlay_threshold, prop5_dir / "tables" / "prop5_overlay_threshold_selection.csv")
    write_csv(overlay_placebo, prop5_dir / "tables" / "prop5_overlay_placebo_summary.csv")
    (prop5_dir / "README.md").write_text(
        "\n".join(
            [
                "# Prop 5 Overlay Exploratory",
                "",
                "This folder is exploratory context only. It is not a final paper claim.",
                "",
                "Overlay formula:",
                "",
                "`overlay = CP + active_flag * (challenger - CP)`",
                "",
                "Top-percentile and condition-label overlays are descriptive unless their thresholds are selected using train-only or prefix-validation logic. See `tables/prop5_overlay_threshold_selection.csv` for provenance flags.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_docs(outdir: Path, tables: dict[str, pd.DataFrame], pred_counts: pd.DataFrame, manifest: pd.DataFrame) -> None:
    prop3 = tables.get("prop3_raw_cp_pm_conditional_summary.csv", pd.DataFrame())
    prop4 = tables.get("prop4_overall_smape_summary.csv", pd.DataFrame())
    overlay = tables.get("prop5_selective_overlay_summary.csv", pd.DataFrame())
    alt = tables.get("prop4_alternative_loss_summary.csv", pd.DataFrame())

    best_overlay = overlay.sort_values("mean_advantage_overall_equal_weight", ascending=False).head(5) if not overlay.empty else pd.DataFrame()
    best_always_on = prop4[prop4["model_name"] != "CP_REPO_FRESH"].sort_values(
        "equal_weight_asset_mean_advantage_vs_CP", ascending=False
    ).head(5) if not prop4.empty and "equal_weight_asset_mean_advantage_vs_CP" in prop4.columns else pd.DataFrame()

    (outdir / "README.md").write_text(
        "\n".join(
            [
                "# PM/CP Research Internal Package",
                "",
                f"Generated: {utc_now()}",
                "",
                "This package consolidates the current internal evidence for PM/path-order tests in the HAR/RV repo.",
                "It is not a final paper draft. Prop 5 overlay work is included only as exploratory context for later review.",
                "",
                "## Source Runs",
                "",
                "- Prop 3 raw PM evidence: `outputs/cp_pm_incremental_tests/`.",
                "- Prop 4 corrected repo-CP ladder: `outputs/cp_repo_ops_hg_actions/28269511971/`.",
                "- Prop 5 selective overlay evidence: `outputs/cp_repo_ops_hg_actions/28269511971/audit_recomputed/switching_overlay_summary.csv`.",
                "- Reproducible overlay test suite: `outputs/pm_selective_overlay_tests/`.",
                "",
                "The package deliberately does not copy large full prediction panels. `source_runs/source_run_manifest.csv` records the source paths.",
                "",
                "## Main Takeaways",
                "",
                "- Prop 1 and Prop 2 are theoretical support statements.",
                "- Prop 3 is empirically supported: raw/global PM is noisy overall, while entry and recent-spike regimes can be positive.",
                "- Prop 4 is not proven globally: always-on PM/shape challengers do not beat repo CP under headline SMAPE.",
                "- Prop 5 is a promising exploratory lead: selective overlays can produce positive equal-weight SMAPE advantage on rare active rows and across most assets, but threshold validation needs manual review before draft use.",
                "- `CP_REPO_LRPM` is diagnostic/non-paper-eligible in the current corrected run because the unregularized design is rank deficient and numerically unstable.",
                "",
                "## Key Files",
                "",
                "- `prop3_raw_pm_diagnostics/`: cleaned Prop 3 diagnostics.",
                "- `prop4_corrected_repo_cp_ladder/`: corrected repo-CP 18-model evidence.",
                "- `prop5_overlay_exploratory/`: exploratory overlay context only.",
                "- `model_guide.md`: compact model-family guide for teammates.",
                "- `test_results_guide.md`: what was tested, what passed, and what remains weak.",
                "- `prop_status_summary.md`: proposition-by-proposition status.",
                "- `tables/prop4_overall_smape_summary.csv`: corrected 18-model always-on ladder.",
                "- `tables/prop5_selective_overlay_summary.csv`: strongest overlay evidence.",
                "- `tables/prop5_selective_overlay_recomputed_from_predictions.csv`: regenerated overlay suite from saved prediction panels.",
                "- `audits/`: fast/slow, feature-purity, ridge, and CP reproduction checks.",
                "- `latex/internal_results_summary.tex`: minimal internal write-up skeleton.",
                "",
                "## Interpretation Guardrails",
                "",
                "- If selective gated overlays beat CP, temporal path shape matters after CP.",
                "- If CP-OPS-HG beats gated OPS-C, PM residue structure adds value beyond local shape.",
                "- If OPS-R fails against random/shuffled residues, PM-specific residue structure is not supported.",
                "- If Haar or recent slope matches OPS-C, the result is generic path shape rather than PM-specific.",
                "- Positive mean advantage with sub-50% win rate is a gain-size effect, not dominance.",
                "- If CP-OPS-HG loses to gated OPS-C, OPS-R is contaminating the final hybrid and should be reported as rejected.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    prop_status = [
        ("Prop 1", "Average-based forecasting models lose temporal order.", "theory-ready", "Short proof note only."),
        (
            "Prop 2",
            "PM is an encoding methodology for lag-position information.",
            "theory-ready with nuance",
            "Do not overclaim full additive reconstruction without interactions/joint CRT cells/nonlinear learners.",
        ),
        (
            "Prop 3",
            "Global/raw PM is noisy as a standalone forecasting model.",
            "empirically supported",
            "Use older CP+PM and corrected repo-CP summaries.",
        ),
        (
            "Prop 4",
            "PM adds value inside average-based forecasting models.",
            "not proven globally",
            "Always-on challengers lose under headline SMAPE; some losses/conditions are positive.",
        ),
        (
            "Prop 5",
            "PM/path-order information adds value as a selective overlay.",
            "empirically promising",
            "Make this the internal draft-v1 direction.",
        ),
    ]
    pd.DataFrame(prop_status, columns=["proposition", "statement", "status", "required_output"]).to_csv(
        outdir / "tables" / "proposition_status_table.csv", index=False
    )
    (outdir / "prop_status_summary.md").write_text(
        "# Proposition Status\n\n"
        + "\n".join(
            f"## {name}\n\nStatement: {statement}\n\nStatus: {status}\n\nRequired output: {output}\n"
            for name, statement, status, output in prop_status
        ),
        encoding="utf-8",
    )

    (outdir / "model_guide.md").write_text(
        "\n".join(
            [
                "# Model Guide",
                "",
                "## CP_REPO_FRESH",
                "The corrected benchmark is `CP_REPO_FRESH = RV/HAR + contig_prime_modulo(vol.copy(), n=22, per_day_normalize=False)`. It uses repo `RV*` and `CP_*` feature columns and is the denominator for the corrected repo-CP ladder.",
                "",
                "## Raw PM and Raw CP+PM",
                "`RAW_PM` tests standalone prime-modulo residue averages from `add_prime_modulo_terms(..., n=22)`. `RAW_CP_REPO_PLUS_PM` adds raw PM to the repo CP controls and replicates the older incremental test pattern: noisy overall, useful in some entry/recent-spike regimes.",
                "",
                "## OPS-R",
                "OPS-R uses centered prime-residue contrasts as an incremental shape layer beyond repo CP. The ridge version is the controlled diagnostic; unregularized residue models are noisy.",
                "",
                "## LRPM",
                "LRPM localizes residue contrasts inside lag neighborhoods. The current unregularized `CP_REPO_LRPM` is rank deficient and diagnostic only. `CP_REPO_RIDGE_LRPM` is the paper-eligible repair path for future runs.",
                "",
                "## Recent Slope, Haar, OPS-C",
                "These are generic path-shape controls. If they match or beat PM-specific residue models, the evidence is path-shape rather than prime-specific.",
                "",
                "## Gated OPS-C and OPS-HG",
                "`CP_REPO_GATED_RIDGE_OPS_C` is the practical cleaned shape model. `CP_REPO_OPS_HG` adds heavily shrunk OPS-R to gated OPS-C. If HG loses to gated OPS-C, the PM-residue addition should be rejected.",
                "",
                "## Placebos",
                "Random residues, shuffled PM, and random gates test whether prime residue structure, true lag order, and the real gate matter.",
                "",
                "## Perturbation Identity",
                "For challenger perturbation `delta_t = yhat_model_t - yhat_CP_t` and CP residual `e_t = y_t - yhat_CP_t`, squared-loss improvement is `2 e_t delta_t - delta_t^2`.",
                "",
                "Interpretation: a PM/shape model helps only when its correction aligns with the CP residual enough to overcome its perturbation penalty.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    alt_note = "Alternative-loss table unavailable."
    if not alt.empty:
        winners = alt[(alt["condition"].astype(str) == "all_rows") & (alt["loss_variant"].astype(str) == "standard")]
        winners = winners[winners["loss_metric"].isin(["MAE", "MSE", "RMSE", "SMAPE", "log_RV_MSE"])]
        alt_note = markdown_table(winners.sort_values("pooled_advantage_cp_minus_model", ascending=False), max_rows=12)

    (outdir / "test_results_guide.md").write_text(
        "\n".join(
            [
                "# Test Results Guide",
                "",
                "## Prop 3",
                "Older CP+PM tests show raw CP+PM loses overall but helps cluster-entry and recent-spike conditions.",
                "",
                "## Prop 4",
                "The corrected 18-model repo-CP ladder does not support an always-on PM claim under headline SMAPE.",
                "",
                "Top always-on challengers by equal-weight SMAPE advantage:",
                "",
                markdown_table(best_always_on),
                "",
                "## Prop 5",
                "Selective overlays are exploratory context, not a final paper claim in this package.",
                "",
                markdown_table(best_overlay),
                "",
                "## Alternative Losses",
                alt_note,
                "",
                "## Audits",
                "- CP reproduction passed in the corrected repo-CP run.",
                "- Strong fast/slow passed for all ladder models on the selected 3-asset, 500-row audit.",
                "- Feature purity passed in the recomputed audit.",
                "- Ridge placement passed: CP/RV controls were unpenalized in ridge models.",
                "- `CP_REPO_LRPM` is unstable and should remain diagnostic/non-paper-eligible.",
                "- Manual conditional `n_assets` was recomputed from per-asset rows in the audit outputs.",
                "",
                "## Research Caution",
                "Always-on challengers lose under SMAPE. Several improve MAE/MSE/RMSE or specific regimes, so the defensible claim is selective overlay value, not global PM dominance.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    pred_counts.to_csv(outdir / "data_manifest.csv", index=False)
    (outdir / "data_manifest.md").write_text(
        "\n".join(
            [
                "# Data Manifest",
                "",
                "The source universe is the same 10 assets used in the existing CP+PM incremental tests.",
                "",
                markdown_table(pred_counts),
                "",
                "Large prediction panels remain in the source run folder and are not duplicated here.",
                "",
                "Source-run inventory:",
                "",
                markdown_table(manifest),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    latex = r"""
\section{Research Setup}
This internal KT note summarizes the cleaned PM/CP evidence carried forward in the consolidated working branch. It is not a final paper draft and does not import anything into \texttt{pre-final}.

\section{Correct CP and PM Construction}
Correct CP:
\[
\texttt{CP\_REPO\_FRESH} = \texttt{RV/HAR} + \texttt{contig\_prime\_modulo(vol.copy(), n=22, per\_day\_normalize=False)}.
\]
Correct PM:
\[
\texttt{PM} = \texttt{add\_prime\_modulo\_terms(..., n=22)}
\]
using the repo minimal-prime construction.

Stale/incorrect benchmark: \texttt{CPB\_B*} four-block lag means are diagnostic only and are not the primary CP benchmark.

\section{Proposition Status}
\begin{tabular}{lll}
\hline
Proposition & Status & Use in draft? \\
\hline
Prop 1 & theoretical proof & foundation \\
Prop 2 & theoretical proof & foundation \\
Prop 3 & empirical diagnostic & probably not main paper \\
Prop 4 & not proven globally & internal discussion \\
Prop 5 & promising exploratory lead & later decision \\
\hline
\end{tabular}

\section{Prop 3: Raw PM Diagnostics}
Raw CP+PM is negative on all observations, positive in cluster-entry and recent-spike regimes, and negative in cluster-exit and older-spike regimes. The minimal interpretation is: raw PM is noisy overall but positive in entry/recent-spike regimes.

\section{Prop 4: Corrected Repo-CP 18-Model Ladder}
The corrected 18-model ladder uses GitHub Actions run \texttt{28269511971}. No always-on challenger beats \texttt{CP\_REPO\_FRESH} under headline SMAPE. Some challengers improve MAE/MSE/RMSE and some conditional regimes are positive, so Prop 4 is not proven globally.

\section{Audits and Validity Checks}
\begin{itemize}
\item CP reproduction passed.
\item Target alignment passed.
\item No-lookahead passed.
\item Stronger fast/slow passed.
\item Feature purity passed.
\item Ridge placement passed.
\item LRPM unstable.
\item Manual \texttt{n\_assets} bug fixed in recomputed tables.
\end{itemize}

\section{Why Always-On Models Failed}
Let
\[
\delta_t = \widehat{y}^{model}_t - \widehat{y}^{CP}_t
\]
and
\[
e_t = y_t - \widehat{y}^{CP}_t.
\]
Under squared loss:
\[
L_{CP} - L_{model} = 2 e_t \delta_t - \delta_t^2.
\]
The extra PM/shape correction only helps when it aligns with CP residuals enough to overcome its own perturbation penalty. Current results show that always-on corrections help in some regimes but hurt elsewhere, especially under SMAPE.

\section{Prop 5 Exploratory Direction}
Selective overlay formula:
\[
\texttt{overlay} = \texttt{CP} + \texttt{active\_flag} \times (\texttt{challenger} - \texttt{CP}).
\]
This is exploratory until train-only threshold validation is finalized.

\section{Open Issues Before Drafting}
Rerun the full ladder if \texttt{CP\_REPO\_RIDGE\_LRPM} should enter result tables; complete train-only threshold validation for overlays; decide whether Prop 5 is draft-facing after manual review; keep stale four-block CP outputs archived as diagnostics only.
""".strip()
    (outdir / "latex" / "internal_results_summary.tex").write_text(latex + "\n", encoding="utf-8")


def build_package(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    prop3_dir = Path(args.prop3_dir)
    prop4_run = Path(args.prop4_run)
    overlay_dir = Path(args.overlay_dir)
    pred_root = Path(args.prediction_root)
    ensure_dirs(outdir)
    manifest = source_manifest(prop3_dir, prop4_run, overlay_dir, outdir)
    tables = make_tables(prop3_dir, prop4_run, overlay_dir, outdir)
    pred_counts = prediction_row_counts(pred_root)
    make_figures(tables, outdir)
    write_docs(outdir, tables, pred_counts, manifest)
    write_prop_subfolders(outdir, tables)
    status = {
        "generated_at_utc": utc_now(),
        "outdir": str(outdir.relative_to(PROJECT_ROOT)),
        "prop3_dir": str(prop3_dir.relative_to(PROJECT_ROOT)),
        "prop4_run": str(prop4_run.relative_to(PROJECT_ROOT)),
        "overlay_dir": str(overlay_dir.relative_to(PROJECT_ROOT)),
        "prediction_root": str(pred_root.relative_to(PROJECT_ROOT)),
        "missing_source_files": manifest.loc[~manifest["exists"], "source_path"].tolist(),
        "all_prediction_files_present": bool(pred_counts["exists"].all()),
    }
    (outdir / "source_runs" / "package_build_metadata.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    marker = outdir / "PACKAGE_READY.txt"
    marker.write_text(f"PM research internal package generated at {utc_now()}\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build PM research internal package")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--prop3-dir", default=str(DEFAULT_PROP3_DIR))
    parser.add_argument("--prop4-run", default=str(DEFAULT_PROP4_RUN))
    parser.add_argument("--overlay-dir", default=str(DEFAULT_OVERLAY_DIR))
    parser.add_argument("--prediction-root", default=str(DEFAULT_PRED_ROOT))
    return parser


def main() -> None:
    build_package(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
