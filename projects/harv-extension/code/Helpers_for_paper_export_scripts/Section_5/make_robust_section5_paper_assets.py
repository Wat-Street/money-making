import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "code" / "paper" / "runs" / "section5_robust_push" / "robust_summary" / "tables"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "paper_assets" / "figures"
TABLE_DIR = OUT_DIR / "paper_assets" / "tables"


def _read(name):
    return pd.read_csv(RUN_DIR / name)


def _fmt(x, digits=2):
    if pd.isna(x):
        return "N/A"
    return f"{float(x):.{digits}f}"


def _pm(x, se=None, digits=2):
    if pd.isna(x):
        return "N/A"
    if se is None or pd.isna(se):
        return _fmt(x, digits)
    return f"{_fmt(x, digits)} $\\pm$ {_fmt(se, digits)}"


def _tex_escape(text):
    return str(text).replace("_", "\\_")


def _model_label(text):
    return str(text).replace("_", "-")


def _family_label(text):
    labels = {
        "CP_FC": "CP",
        "PM_FC": "PM",
        "EQWIN": "Equal",
        "PREFIX": "Prefix",
        "SHIFTWIN": "Shifted",
        "COMP_COPRIME": "Composite",
        "COMP_NONCOPRIME": "Non-coprime",
        "RAND": "RAND",
        "CRS": "CRS",
    }
    return labels.get(str(text), str(text).replace("_", "-"))


def _setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 1.4,
        }
    )


def _save(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def make_feature_count_figure():
    df = _read("robust_feature_count_curve.csv")
    order = [
        ("CP_FC", "CP", "#2f7d32", "-", "o"),
        ("PM_FC", "PM", "#333333", "-", "s"),
        ("EQWIN", "Equal windows", "#777777", "--", "^"),
        ("PREFIX", "Prefix windows", "#999999", ":", "D"),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.15))
    for family, label, color, linestyle, marker in order:
        g = df[df["family"] == family].sort_values("feature_count")
        if g.empty:
            continue
        ax.errorbar(
            g["feature_count"],
            g["delta_smape_pp_mean"],
            yerr=g["delta_smape_pp_se"],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=3.8,
            capsize=2.5,
            elinewidth=0.8,
        )
    ax.axhline(0, color="#222222", linewidth=0.7)
    ax.set_xlabel("Extra features beyond HAR-RV")
    ax.set_ylabel("SMAPE improvement vs HAR-RV (pp)")
    ax.set_xticks([3, 6, 9, 10])
    ax.set_xlim(2.5, 10.5)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5, linestyle=":")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    _save(fig, "Section_5_fig_4_feature_count_curve")


def make_random_controls_figure():
    df = _read("robust_summary_vs_baselines.csv")
    df = df[df["baseline"] == "HAR"].copy()
    fig, ax = plt.subplots(figsize=(5.4, 3.15))

    lines = [
        ("CP_FC", "CP", "#2f7d32", "o"),
        ("PM_FC", "PM", "#333333", "s"),
    ]
    for family, label, color, marker in lines:
        g = df[df["family"] == family].sort_values("feature_count")
        g = g[g["feature_count"].isin([3, 6, 9])]
        ax.errorbar(
            g["feature_count"],
            g["delta_smape_pp_mean"],
            yerr=g["delta_smape_pp_se"],
            label=label,
            color=color,
            marker=marker,
            markersize=3.8,
            capsize=2.5,
            elinewidth=0.8,
        )

    for family, label, color, offset, marker in [
        ("RAND", "Random", "#9a9a9a", -0.10, "x"),
        ("CRS", "Contiguous random", "#b5b5b5", 0.10, "+"),
    ]:
        g = df[df["family"] == family]
        for k in [3, 6, 9]:
            vals = g[g["feature_count"] == k]["delta_smape_pp_mean"].astype(float)
            if vals.empty:
                continue
            ax.scatter(
                np.repeat(k + offset, len(vals)),
                vals,
                label=label if k == 3 else None,
                color=color,
                marker=marker,
                s=24,
                linewidths=0.9,
            )
    ax.axhline(0, color="#222222", linewidth=0.7)
    ax.set_xlabel("Extra features beyond HAR-RV")
    ax.set_ylabel("SMAPE improvement vs HAR-RV (pp)")
    ax.set_xticks([3, 6, 9])
    ax.set_xlim(2.5, 9.5)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5, linestyle=":")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    _save(fig, "Section_5_fig_5_random_controls")


def _summary_lookup():
    metrics = _read("robust_summary_by_model.csv")
    comp = _read("robust_summary_vs_baselines.csv")
    comp = comp[comp["baseline"] == "HAR"].copy()
    by_model = {}
    for _, row in metrics.iterrows():
        by_model[row["model"]] = row.to_dict()
    by_delta = {}
    for _, row in comp.iterrows():
        by_delta[row["model"]] = row.to_dict()
    return by_model, by_delta


def _model_row(label, model, extra_features, by_model, by_delta):
    m = by_model.get(model, {})
    d = by_delta.get(model, {})
    smape = _pm(m.get("smape_pct_mean"), m.get("smape_pct_se"))
    if model == "HAR":
        delta = "N/A"
        win = "N/A"
    else:
        delta = _pm(d.get("delta_smape_pp_mean"), d.get("delta_smape_pp_se"))
        win = _fmt(d.get("win_rate_pct_mean"), 1)
    return [label, extra_features, smape, delta, win]


def write_main_summary_table():
    by_model, by_delta = _summary_lookup()
    rows = [
        _model_row("HAR-RV", "HAR", "0", by_model, by_delta),
        _model_row("Prime Modulo (PM)", "PM", "10", by_model, by_delta),
        _model_row("Contiguous Prime (CP)", "CP", "10", by_model, by_delta),
        _model_row("PM, 6 features", "PM_FC6", "6", by_model, by_delta),
        _model_row("CP, 6 features", "CP_FC6", "6", by_model, by_delta),
        _model_row("Equal windows, 6 features", "EQWIN_FC6", "6", by_model, by_delta),
        _model_row("Prefix windows, 6 features", "PREFIX_FC6", "6", by_model, by_delta),
    ]
    body = "\n".join(
        f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} \\\\"
        for r in rows
    )
    tex = rf"""\begin{{tabular}}{{|l|c|c|c|c|}}
\hline
Model & Extra Features & SMAPE (\%) & \shortstack{{$\Delta$SMAPE\\vs HAR-RV (pp)}} & \shortstack{{Timestamp\\Win Rate (\%)}} \\
\hline
{body}
\hline
\end{{tabular}}"""
    (TABLE_DIR / "Section_5_table_3_robust_summary.tex").write_text(tex, encoding="utf-8")


def _aggregate_random(family, k):
    comp = _read("robust_summary_vs_baselines.csv")
    g = comp[(comp["baseline"] == "HAR") & (comp["family"] == family) & (comp["feature_count"] == k)]
    vals = g["delta_smape_pp_mean"].astype(float)
    wins = g["win_rate_pct_mean"].astype(float)
    if vals.empty:
        return "N/A", "N/A"
    se = vals.std(ddof=1) / math.sqrt(len(vals)) if len(vals) > 1 else np.nan
    return _pm(vals.mean(), se), _fmt(wins.mean(), 1)


def write_driver_table():
    _, by_delta = _summary_lookup()
    rows = [
        ["CP, 6 features", "Local segmentation", _pm(by_delta["CP_FC6"]["delta_smape_pp_mean"], by_delta["CP_FC6"]["delta_smape_pp_se"]), _fmt(by_delta["CP_FC6"]["win_rate_pct_mean"], 1)],
        ["CP, 9 features", "Local segmentation", _pm(by_delta["CP_FC9"]["delta_smape_pp_mean"], by_delta["CP_FC9"]["delta_smape_pp_se"]), _fmt(by_delta["CP_FC9"]["win_rate_pct_mean"], 1)],
        ["PM, 6 features", "Prime modulo", _pm(by_delta["PM_FC6"]["delta_smape_pp_mean"], by_delta["PM_FC6"]["delta_smape_pp_se"]), _fmt(by_delta["PM_FC6"]["win_rate_pct_mean"], 1)],
        ["Composite modulo, 6 features", "Non-prime modulo", _pm(by_delta["COMP_COPRIME_FC6"]["delta_smape_pp_mean"], by_delta["COMP_COPRIME_FC6"]["delta_smape_pp_se"]), _fmt(by_delta["COMP_COPRIME_FC6"]["win_rate_pct_mean"], 1)],
        ["Non-coprime modulo, 6 features", "Non-prime modulo", _pm(by_delta["COMP_NONCOPRIME_FC6"]["delta_smape_pp_mean"], by_delta["COMP_NONCOPRIME_FC6"]["delta_smape_pp_se"]), _fmt(by_delta["COMP_NONCOPRIME_FC6"]["win_rate_pct_mean"], 1)],
    ]
    rand_delta, rand_win = _aggregate_random("RAND", 6)
    crs_delta, crs_win = _aggregate_random("CRS", 6)
    rows.extend(
        [
            ["Random controls, 6 features", "Random sets", rand_delta, rand_win],
            ["Contiguous random, 6 features", "Random local sets", crs_delta, crs_win],
            ["HARQ", "External HAR variant", _pm(by_delta["HARQ"]["delta_smape_pp_mean"], by_delta["HARQ"]["delta_smape_pp_se"]), _fmt(by_delta["HARQ"]["win_rate_pct_mean"], 1)],
            ["HAR-TCJ", "External HAR variant", _pm(by_delta["HAR_TCJ"]["delta_smape_pp_mean"], by_delta["HAR_TCJ"]["delta_smape_pp_se"]), _fmt(by_delta["HAR_TCJ"]["win_rate_pct_mean"], 1)],
            ["EWMA", "External baseline", _pm(by_delta["EWMA"]["delta_smape_pp_mean"], by_delta["EWMA"]["delta_smape_pp_se"]), _fmt(by_delta["EWMA"]["win_rate_pct_mean"], 1)],
            ["GARCH(1,1)", "External baseline", _pm(by_delta["GARCH11"]["delta_smape_pp_mean"], by_delta["GARCH11"]["delta_smape_pp_se"]), _fmt(by_delta["GARCH11"]["win_rate_pct_mean"], 1)],
        ]
    )
    body = "\n".join(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} \\\\" for r in rows)
    tex = rf"""\begin{{tabular}}{{|l|l|c|c|}}
\hline
Comparison & Driver Tested & \shortstack{{$\Delta$SMAPE\\vs HAR-RV (pp)}} & \shortstack{{Timestamp\\Win Rate (\%)}} \\
\hline
{body}
\hline
\end{{tabular}}"""
    (TABLE_DIR / "Section_5_table_4_driver_checks.tex").write_text(tex, encoding="utf-8")


def write_appendix_table(source_name, out_name, columns, label_map, digits=None):
    digits = digits or {}
    df = _read(source_name)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            val = row[col]
            if col in label_map:
                cells.append(label_map[col](val, row))
            elif isinstance(val, (float, np.floating)):
                cells.append(_fmt(val, digits.get(col, 2)))
            else:
                cells.append(_tex_escape(val))
        rows.append(" & ".join(cells) + r" \\")
    colspec = "|" + "|".join(["l"] + ["c"] * (len(columns) - 1)) + "|"
    header = " & ".join(label_map.get(f"header_{col}", lambda v, r=None: _tex_escape(col))(col) for col in columns)
    tex = rf"""\begin{{tabular}}{{{colspec}}}
\hline
{header} \\
\hline
{chr(10).join(rows)}
\hline
\end{{tabular}}"""
    (TABLE_DIR / out_name).write_text(tex, encoding="utf-8")


def write_appendix_tables():
    label_map = {
        "model": lambda v, r: _tex_escape(_model_label(v)),
        "family": lambda v, r: _tex_escape(_family_label(v)),
        "header_model": lambda v, r=None: "Model",
        "header_family": lambda v, r=None: "Family",
        "header_feature_count": lambda v, r=None: "Features",
        "header_delta_smape_pp_mean": lambda v, r=None: "$\\Delta$SMAPE",
        "header_delta_smape_pp_se": lambda v, r=None: "SE",
        "header_win_rate_pct_mean": lambda v, r=None: "Win Rate (\\%)",
        "header_n_tickers": lambda v, r=None: "Assets",
        "header_smape_pct_mean": lambda v, r=None: "SMAPE",
        "header_smape_pct_se": lambda v, r=None: "SE",
    }
    appendix_digits = {
        "feature_count": 0,
        "delta_smape_pp_mean": 2,
        "delta_smape_pp_se": 2,
        "win_rate_pct_mean": 1,
        "n_tickers": 0,
    }
    write_appendix_table(
        "robust_feature_count_curve.csv",
        "Appendix_Table_C1_feature_count.tex",
        ["model", "family", "feature_count", "delta_smape_pp_mean", "delta_smape_pp_se", "win_rate_pct_mean"],
        label_map,
        appendix_digits,
    )
    write_appendix_table(
        "robust_contiguous_windows_summary.csv",
        "Appendix_Table_C2_contiguous_windows.tex",
        ["model", "family", "feature_count", "delta_smape_pp_mean", "delta_smape_pp_se", "win_rate_pct_mean"],
        label_map,
        appendix_digits,
    )
    write_appendix_table(
        "robust_prime_vs_nonprime_summary.csv",
        "Appendix_Table_C3_prime_nonprime.tex",
        ["model", "family", "feature_count", "delta_smape_pp_mean", "delta_smape_pp_se", "win_rate_pct_mean"],
        label_map,
        appendix_digits,
    )
    write_appendix_table(
        "robust_random_controls_summary.csv",
        "Appendix_Table_C4_random_controls.tex",
        ["model", "family", "feature_count", "delta_smape_pp_mean", "delta_smape_pp_se", "win_rate_pct_mean"],
        label_map,
        appendix_digits,
    )
    external = _read("robust_summary_vs_baselines.csv")
    external = external[(external["baseline"] == "HAR") & external["model"].isin(["HARQ", "HAR_TCJ", "EWMA", "GARCH11"])].copy()
    external.to_csv(TABLE_DIR / "Appendix_Table_C5_external_source.csv", index=False)
    rows = []
    for _, row in external.sort_values("delta_smape_pp_mean", ascending=False).iterrows():
        rows.append(
            f"{_tex_escape(row['model'])} & {_pm(row['delta_smape_pp_mean'], row['delta_smape_pp_se'])} & "
            f"{_fmt(row['win_rate_pct_mean'], 1)} & {_fmt(row['n_tickers'], 0)} \\\\"
        )
    tex = rf"""\begin{{tabular}}{{|l|c|c|c|}}
\hline
Model & \shortstack{{$\Delta$SMAPE\\vs HAR-RV (pp)}} & \shortstack{{Timestamp\\Win Rate (\%)}} & Assets \\
\hline
{chr(10).join(rows)}
\hline
\end{{tabular}}"""
    (TABLE_DIR / "Appendix_Table_C5_external.tex").write_text(tex, encoding="utf-8")


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    _setup_style()
    make_feature_count_figure()
    make_random_controls_figure()
    write_main_summary_table()
    write_driver_table()
    write_appendix_tables()


if __name__ == "__main__":
    main()
