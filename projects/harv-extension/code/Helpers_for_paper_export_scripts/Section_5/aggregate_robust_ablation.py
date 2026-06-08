import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


LOWER_IS_BETTER = {"smape_pct", "rmse", "mae", "medae"}


def _parse_model(model):
    out = {
        "family": model,
        "feature_count": np.nan,
        "seed": np.nan,
        "offset": np.nan,
        "suite": "",
    }
    fc = re.search(r"_FC(\d+)", model)
    seed = re.search(r"_S(\d+)", model)
    offset = re.search(r"_O(\d+)", model)
    if fc:
        out["feature_count"] = int(fc.group(1))
    if seed:
        out["seed"] = int(seed.group(1))
    if offset:
        out["offset"] = int(offset.group(1))

    if model in {"HAR", "PM", "CP"}:
        out["family"] = model
        out["suite"] = "core"
        if model in {"PM", "CP"}:
            out["feature_count"] = 10
        elif model == "HAR":
            out["feature_count"] = 0
    elif model.startswith("PM_FC"):
        out["family"] = "PM_FC"
        out["suite"] = "capacity"
    elif model.startswith("CP_FC"):
        out["family"] = "CP_FC"
        out["suite"] = "capacity"
    elif model.startswith("EQWIN"):
        out["family"] = "EQWIN"
        out["suite"] = "capacity"
    elif model.startswith("PREFIX"):
        out["family"] = "PREFIX"
        out["suite"] = "capacity"
    elif model.startswith("SHIFTWIN"):
        out["family"] = "SHIFTWIN"
        out["suite"] = "contiguous"
    elif model.startswith("COMP_COPRIME"):
        out["family"] = "COMP_COPRIME"
        out["suite"] = "modulo"
    elif model.startswith("COMP_NONCOPRIME"):
        out["family"] = "COMP_NONCOPRIME"
        out["suite"] = "modulo"
    elif model.startswith("RAND"):
        out["family"] = "RAND"
        out["suite"] = "random"
    elif model.startswith("CRS"):
        out["family"] = "CRS"
        out["suite"] = "random"
    elif model in {"EWMA", "GARCH11", "HARQ", "HAR_TCJ"}:
        out["family"] = model
        out["suite"] = "external"
        if model == "HARQ":
            out["feature_count"] = 4
    return out


def _read_prediction_file(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    if "Date" not in df.columns:
        raise ValueError(f"{path} does not contain Date")
    return df.set_index("Date")


def merge_predictions(pred_dir, merged_dir):
    pred_dir = Path(pred_dir)
    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    by_ticker = {}
    for path in pred_dir.glob("*.csv"):
        stem = path.stem
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        ticker, suite = parts
        by_ticker.setdefault(ticker, []).append(path)

    merged = {}
    for ticker, paths in sorted(by_ticker.items()):
        acc = None
        for path in sorted(paths):
            df = _read_prediction_file(path)
            if acc is None:
                acc = df
                continue
            actual = df["Actual"].copy() if "Actual" in df.columns else None
            if "Actual" in acc.columns and "Actual" in df.columns:
                df = df.drop(columns=["Actual"])
            duplicate_cols = [c for c in df.columns if c in acc.columns]
            if duplicate_cols:
                acc = acc.drop(columns=duplicate_cols)
            acc = acc.join(df, how="outer")
            if actual is not None and "Actual" in acc.columns:
                acc["Actual"] = acc["Actual"].combine_first(actual.reindex(acc.index))
        if acc is not None:
            out = merged_dir / f"{ticker}.csv"
            acc.sort_index().reset_index().rename(columns={"index": "Date"}).to_csv(out, index=False)
            merged[ticker] = acc.sort_index()
    return merged


def _metric_frame(ticker, df):
    rows = []
    actual = df["Actual"]
    for col in [c for c in df.columns if c.startswith("Predicted_")]:
        model = col.replace("Predicted_", "")
        pred = df[col]
        mask = actual.notna() & pred.notna()
        if mask.sum() < 5:
            continue
        y = actual[mask].astype(float)
        p = pred[mask].astype(float)
        err = y - p
        smape = 200.0 * np.abs(err) / np.maximum(1e-12, np.abs(y) + np.abs(p))
        prev_actual = y.shift(1)
        da_mask = prev_actual.notna()
        if da_mask.sum() > 0:
            da = (
                np.sign(y[da_mask] - prev_actual[da_mask])
                == np.sign(p[da_mask] - prev_actual[da_mask])
            ).mean() * 100.0
        else:
            da = np.nan
        meta = _parse_model(model)
        rows.append(
            {
                "ticker": ticker,
                "model": model,
                **meta,
                "n_obs": int(mask.sum()),
                "smape_pct": float(smape.mean()),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mae": float(np.mean(np.abs(err))),
                "medae": float(np.median(np.abs(err))),
                "diracc_pct": float(da),
            }
        )
    return pd.DataFrame(rows)


def _newey_west_pvalue(d):
    d = pd.Series(d).dropna().astype(float).to_numpy()
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    centered = d - d.mean()
    lag = max(1, int(round(4 * (n / 100.0) ** (2.0 / 9.0))))
    gamma0 = float(np.dot(centered, centered) / n)
    lrv = gamma0
    for ell in range(1, min(lag, n - 1) + 1):
        gamma = float(np.dot(centered[ell:], centered[:-ell]) / n)
        weight = 1.0 - ell / (lag + 1.0)
        lrv += 2.0 * weight * gamma
    se = np.sqrt(max(lrv, 1e-18) / n)
    stat = float(d.mean() / se)
    pval = float(2.0 * stats.norm.sf(abs(stat)))
    return stat, pval


def _comparison_frame(ticker, df, baselines=("HAR", "PM", "CP")):
    rows = []
    actual = df["Actual"]
    pred_cols = {c.replace("Predicted_", ""): c for c in df.columns if c.startswith("Predicted_")}
    for baseline in baselines:
        if baseline not in pred_cols:
            continue
        base_pred = df[pred_cols[baseline]]
        for model, model_col in pred_cols.items():
            if model == baseline:
                continue
            pred = df[model_col]
            mask = actual.notna() & base_pred.notna() & pred.notna()
            if mask.sum() < 10:
                continue
            y = actual[mask].astype(float)
            b = base_pred[mask].astype(float)
            p = pred[mask].astype(float)
            base_abs = np.abs(y - b)
            model_abs = np.abs(y - p)
            base_smape = 200.0 * base_abs / np.maximum(1e-12, np.abs(y) + np.abs(b))
            model_smape = 200.0 * model_abs / np.maximum(1e-12, np.abs(y) + np.abs(p))
            d_abs = base_abs - model_abs
            d_smape = base_smape - model_smape
            dm_abs, p_abs = _newey_west_pvalue(d_abs)
            dm_smape, p_smape = _newey_west_pvalue(d_smape)
            meta = _parse_model(model)
            rows.append(
                {
                    "ticker": ticker,
                    "baseline": baseline,
                    "model": model,
                    **meta,
                    "n_obs": int(mask.sum()),
                    "delta_smape_pp": float(d_smape.mean()),
                    "delta_mae": float(d_abs.mean()),
                    "win_rate_pct": float((d_smape > 0).mean() * 100.0),
                    "dm_abs_stat": dm_abs,
                    "dm_abs_p": p_abs,
                    "dm_smape_stat": dm_smape,
                    "dm_smape_p": p_smape,
                }
            )
    return pd.DataFrame(rows)


def _summarize(df, group_cols, value_cols):
    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["n_tickers"] = group["ticker"].nunique() if "ticker" in group.columns else len(group)
        for col in value_cols:
            vals = group[col].dropna().astype(float)
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_se"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_capacity(comparisons, outdir):
    frame = _capacity_frame(comparisons)
    if frame.empty:
        return
    summary = _summarize(frame, ["family", "feature_count"], ["delta_smape_pp"])
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for family, group in summary.groupby("family"):
        group = group.sort_values("feature_count")
        ax.errorbar(
            group["feature_count"],
            group["delta_smape_pp_mean"],
            yerr=group["delta_smape_pp_se"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=family,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Extra features beyond HAR")
    ax.set_ylabel("SMAPE improvement vs HAR (pp)")
    ax.set_title("Feature Count Curve")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(Path(outdir) / f"robust_feature_count_curve.{ext}", dpi=220)
    plt.close(fig)


def _capacity_frame(comparisons):
    frame = comparisons[
        (comparisons["baseline"] == "HAR")
        & comparisons["family"].isin(["PM_FC", "CP_FC", "EQWIN", "PREFIX"])
    ].copy()
    full = comparisons[
        (comparisons["baseline"] == "HAR")
        & comparisons["model"].isin(["PM", "CP"])
    ].copy()
    if not full.empty:
        full["family"] = full["model"].map({"PM": "PM_FC", "CP": "CP_FC"})
        full["feature_count"] = 10
        full["suite"] = "capacity"
        frame = pd.concat([frame, full], ignore_index=True)
    return frame


def _plot_family_bars(comparisons, outdir):
    frame = comparisons[comparisons["baseline"] == "HAR"].copy()
    keep = [
        "PM",
        "CP",
        "PM_FC",
        "CP_FC",
        "EQWIN",
        "PREFIX",
        "SHIFTWIN",
        "COMP_COPRIME",
        "COMP_NONCOPRIME",
        "RAND",
        "CRS",
        "EWMA",
        "GARCH11",
        "HARQ",
        "HAR_TCJ",
    ]
    frame = frame[frame["family"].isin(keep)]
    if frame.empty:
        return
    summary = _summarize(frame, ["family"], ["delta_smape_pp"])
    summary = summary.sort_values("delta_smape_pp_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar(summary["family"], summary["delta_smape_pp_mean"], yerr=summary["delta_smape_pp_se"], capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("SMAPE improvement vs HAR (pp)")
    ax.set_title("Robust Ablation Families")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(Path(outdir) / f"robust_family_summary.{ext}", dpi=220)
    plt.close(fig)


def _plot_external(metrics, outdir):
    models = ["HAR", "PM", "CP", "EWMA", "GARCH11", "HARQ", "HAR_TCJ"]
    frame = metrics[metrics["model"].isin(models)].copy()
    if frame.empty:
        return
    summary = _summarize(frame, ["model"], ["smape_pct", "rmse", "mae", "diracc_pct"])
    summary["order"] = summary["model"].map({m: i for i, m in enumerate(models)})
    summary = summary.sort_values("order")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(summary["model"], summary["smape_pct_mean"], yerr=summary["smape_pct_se"], capsize=3)
    ax.set_ylabel("SMAPE (%)")
    ax.set_title("External Baseline Comparison")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(Path(outdir) / f"robust_external_baselines.{ext}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Aggregate robust Section 5 ablation artifacts")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    merged_dir = outdir / "merged_predictions"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    merged = merge_predictions(args.pred_dir, merged_dir)
    metric_parts = []
    comparison_parts = []
    for ticker, df in merged.items():
        metric_parts.append(_metric_frame(ticker, df))
        comparison_parts.append(_comparison_frame(ticker, df))

    metrics = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    comparisons = pd.concat(comparison_parts, ignore_index=True) if comparison_parts else pd.DataFrame()

    metrics.to_csv(tables_dir / "robust_metrics_by_ticker.csv", index=False)
    comparisons.to_csv(tables_dir / "robust_comparisons_by_ticker.csv", index=False)

    if not metrics.empty:
        metric_summary = _summarize(
            metrics,
            ["model", "family", "feature_count", "suite"],
            ["smape_pct", "rmse", "mae", "medae", "diracc_pct"],
        )
        metric_summary.to_csv(tables_dir / "robust_summary_by_model.csv", index=False)
        _plot_external(metrics, figures_dir)

    if not comparisons.empty:
        comparison_summary = _summarize(
            comparisons,
            ["baseline", "model", "family", "feature_count", "suite"],
            ["delta_smape_pp", "delta_mae", "win_rate_pct"],
        )
        comparison_summary.to_csv(tables_dir / "robust_summary_vs_baselines.csv", index=False)
        capacity = _capacity_frame(comparison_summary)
        capacity.to_csv(tables_dir / "robust_feature_count_curve.csv", index=False)
        random_controls = comparison_summary[
            (comparison_summary["baseline"] == "HAR")
            & (comparison_summary["family"].isin(["RAND", "CRS", "PM_FC", "CP_FC"]))
        ].copy()
        random_controls.to_csv(tables_dir / "robust_random_controls_summary.csv", index=False)
        modulo = comparison_summary[
            (comparison_summary["baseline"] == "HAR")
            & (comparison_summary["family"].isin(["PM_FC", "COMP_COPRIME", "COMP_NONCOPRIME"]))
        ].copy()
        modulo.to_csv(tables_dir / "robust_prime_vs_nonprime_summary.csv", index=False)
        contiguous = comparison_summary[
            (comparison_summary["baseline"] == "HAR")
            & (comparison_summary["family"].isin(["CP_FC", "EQWIN", "PREFIX", "SHIFTWIN"]))
        ].copy()
        contiguous.to_csv(tables_dir / "robust_contiguous_windows_summary.csv", index=False)
        _plot_capacity(comparisons, figures_dir)
        _plot_family_bars(comparisons, figures_dir)

    print(f"Wrote robust ablation package to {outdir}")


if __name__ == "__main__":
    main()
