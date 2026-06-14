from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "final_paper" / "assets" / "figures_journal"
TABLES = ROOT / "run_results" / "paper_runs" / "test_7" / "tables"
PRED = ROOT / "run_results" / "paper_runs" / "test_7" / "predictions"
ROBUST = ROOT / "run_results" / "paper_runs" / "section5_robust_push" / "robust_summary" / "tables"

COLOR_HAR = "#222222"
COLOR_PM = "#5f6f7a"
COLOR_CP = "#2f6f4e"
COLOR_REF = "#a8a8a8"
COLOR_LIGHT = "#d8d8d8"
COLOR_POS = "#2f6f4e"
COLOR_NEG = "#9b4d4d"


def setup_style():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.35,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
        }
    )


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.55, color="#d2d2d2")
    ax.set_axisbelow(True)


def section2_figures():
    df = pd.read_csv(TABLES / "Section_2_regimes.csv")
    df["model"] = pd.Categorical(df["model"], ["HAR", "PM", "CP"], ordered=True)
    df = df.sort_values("model")
    labels = ["HAR-RV", "PM", "CP"]
    x = np.arange(len(df))
    width = 0.23
    colors = {"Low": "#3f3f3f", "Medium": "#858585", "High": "#c8c8c8"}

    fig, ax = plt.subplots(figsize=(4.25, 2.55))
    ax.bar(x - width, df["smape_low_pct_mean"], width, yerr=df["smape_low_pct_se"], capsize=2.5, color=colors["Low"], label="Low")
    ax.bar(x, df["smape_med_pct_mean"], width, yerr=df["smape_med_pct_se"], capsize=2.5, color=colors["Medium"], label="Medium")
    ax.bar(x + width, df["smape_high_pct_mean"], width, yerr=df["smape_high_pct_se"], capsize=2.5, color=colors["High"], label="High")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"SMAPE (\%)")
    ax.set_ylim(0, 190)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), columnspacing=1.0, handlelength=1.2)
    style_axis(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "Section_2_fig_1_regimes_absolute")

    har = df[df["model"] == "HAR"].iloc[0]
    rel = df[df["model"] != "HAR"].copy()
    rel_labels = ["PM", "CP"]
    x = np.arange(len(rel))
    fig, ax = plt.subplots(figsize=(4.25, 2.55))
    regime_labels = {"low": "Low", "med": "Medium", "high": "High"}
    for idx, (regime, pos, color) in enumerate(
        [
            ("low", -width, colors["Low"]),
            ("med", 0, colors["Medium"]),
            ("high", width, colors["High"]),
        ]
    ):
        mean_col = f"smape_{regime}_pct_mean"
        se_col = f"smape_{regime}_pct_se"
        h_mean = float(har[mean_col])
        h_se = float(har[se_col])
        means = 100.0 * (h_mean - rel[mean_col].astype(float)) / h_mean
        ses = 100.0 * np.sqrt((rel[se_col].astype(float) ** 2) / (h_mean**2) + ((rel[mean_col].astype(float) ** 2) * (h_se**2) / (h_mean**4)))
        ax.bar(x + pos, means, width, yerr=ses, capsize=2.5, color=color, label=regime_labels[regime])
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(rel_labels)
    ax.set_ylabel(r"Improvement vs HAR-RV (\%)")
    ax.set_ylim(-19, 19)
    ax.set_yticks([-15, -10, -5, 0, 5, 10, 15])
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), columnspacing=1.0, handlelength=1.2)
    style_axis(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "Section_2_fig_1_regimes_relative")


def load_all_mean():
    df = pd.read_csv(PRED / "combined" / "ALL_MEAN.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


def smape(actual, pred):
    denom = np.maximum(1e-12, np.abs(actual) + np.abs(pred))
    return 200.0 * np.abs(actual - pred) / denom


def section3_temporal():
    df = load_all_mean()
    fig, ax = plt.subplots(figsize=(6.6, 3.05))
    styles = {
        "HAR": ("HAR-RV", COLOR_HAR, "-"),
        "PM": ("Prime Modulo (PM)", COLOR_PM, "--"),
        "CP": ("Contiguous Prime (CP)", COLOR_CP, "-."),
    }
    for model, (label, color, linestyle) in styles.items():
        s = smape(df["Actual"], df[f"Predicted_{model}"]).rolling(window=78, min_periods=19).mean()
        s = s.ewm(span=39, adjust=False).mean().resample("W").mean().dropna()
        ax.plot(s.index, s.values, color=color, linestyle=linestyle, label=label)
    ax.set_ylabel(r"Rolling SMAPE (\%)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(r"%Y-%m"))
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14), columnspacing=1.2, handlelength=2.2)
    style_axis(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "Section_3_fig_2_temporal_stability_ALL_MEAN")


def pvalue_text(_p):
    return r"$<10^{-6}$"


def section4_advantage():
    df = load_all_mean()
    y = df["Actual"].astype(float)
    base = smape(y, df["Predicted_HAR"].astype(float))

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 3.9), sharex=True, sharey=True)
    for ax, model, title in zip(axes, ["PM", "CP"], ["Prime Modulo (PM)", "Contiguous Prime (CP)"]):
        delta = base - smape(y, df[f"Predicted_{model}"].astype(float))
        plot_s = pd.Series(delta.values, index=df.index).resample("W").median()
        line_s = plot_s.ewm(span=7, adjust=False).mean()
        ax.axhline(0, color=COLOR_HAR, linewidth=0.8)
        ax.fill_between(plot_s.index, 0, np.where(plot_s.values > 0, plot_s.values, 0), color=COLOR_POS, alpha=0.26, label=f"{model} better")
        ax.fill_between(plot_s.index, 0, np.where(plot_s.values < 0, plot_s.values, 0), color=COLOR_NEG, alpha=0.20, label="HAR-RV better")
        ax.plot(line_s.index, line_s.values, color=COLOR_HAR, linewidth=0.95)
        ax.text(0.01, 0.90, title, transform=ax.transAxes, ha="left", va="top")
        ax.legend(frameon=False, loc="upper right", ncol=2, columnspacing=1.2, handlelength=1.5)
        style_axis(ax)
    axes[1].set_xlabel("Date")
    for ax in axes:
        ax.set_ylim(-11, 16)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter(r"%Y-%m"))
    fig.text(0.02, 0.5, r"$\Delta$SMAPE vs HAR-RV (pp)", rotation=90, va="center")
    fig.tight_layout(rect=(0.04, 0, 1, 1))
    save(fig, "Section_4_fig_3_error_advantage_ALL_MEAN")


def section5_feature_count():
    df = pd.read_csv(ROBUST / "robust_feature_count_curve.csv")
    order = [
        ("CP_FC", "CP", COLOR_CP, "-", "o"),
        ("PM_FC", "PM", COLOR_PM, "--", "s"),
        ("EQWIN", "Equal windows", "#6b6b6b", "-.", "^"),
        ("PREFIX", "Prefix windows", COLOR_REF, ":", "D"),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.35))
    for family, label, color, linestyle, marker in order:
        g = df[df["family"] == family].sort_values("feature_count")
        if g.empty:
            continue
        ax.errorbar(g["feature_count"], g["delta_smape_pp_mean"], yerr=g["delta_smape_pp_se"], label=label, color=color, linestyle=linestyle, marker=marker, markersize=4.0, capsize=2.5, elinewidth=0.8)
    ax.axhline(0, color=COLOR_HAR, linewidth=0.7)
    ax.set_xlabel("Extra features beyond HAR-RV")
    ax.set_ylabel("SMAPE improvement vs HAR-RV (pp)")
    ax.set_xticks([3, 6, 9, 10])
    ax.set_xlim(2.5, 10.5)
    ax.set_ylim(-0.05, 1.8)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), columnspacing=1.0, handlelength=1.9)
    style_axis(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "Section_5_fig_4_feature_count_curve")


def section5_random_controls():
    df = pd.read_csv(ROBUST / "robust_summary_vs_baselines.csv")
    df = df[df["baseline"] == "HAR"].copy()
    fig, ax = plt.subplots(figsize=(6.0, 3.35))
    for family, label, color, linestyle, marker in [("CP_FC", "CP", COLOR_CP, "-", "o"), ("PM_FC", "PM", COLOR_PM, "--", "s")]:
        g = df[(df["family"] == family) & (df["feature_count"].isin([3, 6, 9]))].sort_values("feature_count")
        ax.errorbar(g["feature_count"], g["delta_smape_pp_mean"], yerr=g["delta_smape_pp_se"], label=label, color=color, linestyle=linestyle, marker=marker, markersize=4.0, capsize=2.5, elinewidth=0.8)
    for family, label, color, offset, marker in [
        ("RAND", "Random", "#737373", -0.10, "x"),
        ("CRS", "Contiguous random", COLOR_REF, 0.10, "+"),
    ]:
        g = df[df["family"] == family]
        for k in [3, 6, 9]:
            vals = g[g["feature_count"] == k]["delta_smape_pp_mean"].astype(float)
            ax.scatter(np.repeat(k + offset, len(vals)), vals, label=label if k == 3 else None, color=color, marker=marker, s=30, linewidths=1.0)
    ax.axhline(0, color=COLOR_HAR, linewidth=0.7)
    ax.set_xlabel("Extra features beyond HAR-RV")
    ax.set_ylabel("SMAPE improvement vs HAR-RV (pp)")
    ax.set_xticks([3, 6, 9])
    ax.set_xlim(2.5, 9.5)
    ax.set_ylim(-0.15, 1.8)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), columnspacing=1.0, handlelength=1.9)
    style_axis(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "Section_5_fig_5_random_controls")


def section6_asset_groups():
    df = pd.read_csv(TABLES / "Section_6_groups_pivot_long.csv")
    order = ["US Mega Tech", "Broad US Equity", "International Equity", "Commodities/Metals", "Fixed Income"]
    labels = ["US Mega\nTech", "Broad US\nEquity", "International\nEquity", "Commodities/\nMetals", "Fixed\nIncome"]
    x = np.arange(len(order))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    for offset, model, color in [(-width / 2, "PM", COLOR_PM), (width / 2, "CP", COLOR_CP)]:
        vals = []
        ses = []
        for group in order:
            row = df[(df["group"] == group) & (df["model"] == model)].iloc[0]
            vals.append(float(row["delta_smape_pct_mean"]))
            ses.append(float(row["delta_smape_pct_se"]) if pd.notna(row["delta_smape_pct_se"]) else 0.0)
        ax.bar(x + offset, vals, width, yerr=ses, capsize=2.5, label=model, color=color)
    ax.axhline(0, color=COLOR_HAR, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"SMAPE improvement vs HAR-RV (pp)")
    ax.set_ylim(-0.35, 3.35)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16), columnspacing=1.2)
    style_axis(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "Section_6_fig_7_asset_groups")


def main():
    setup_style()
    section2_figures()
    section3_temporal()
    section4_advantage()
    section5_feature_count()
    section5_random_controls()
    section6_asset_groups()


if __name__ == "__main__":
    main()
