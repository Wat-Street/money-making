import os
import re
import sys
import json
import shutil
import subprocess
from datetime import datetime
from typing import List

import pandas as pd
import shlex


ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
COMMANDS_MD = os.path.join(PROJECT, "config", "result_generation_commands.md")
RUNS_DIR = os.path.join(PROJECT, "run_results", "paper_runs")
LATEST_META = os.path.join(RUNS_DIR, "latest.json")

DEFAULT_TICKERS_10 = "SPY,AAPL,NVDA,IWM,EEM,GLD,TLT,USO,ARKK,VXX"
DEFAULT_MODELS = "HAR,PM,CP"


def _prompt(msg, default=None):
    suffix = f" [{default}]" if default else ""
    val = input(f"{msg}{suffix}: ").strip()
    return val if val else default


def _prompt_bool(msg, default=True):
    hint = "Y/n" if default else "y/N"
    val = input(f"{msg} ({hint}): ").strip().lower()
    if not val:
        return default
    return val in {"y", "yes", "1", "true"}


def _split_csv(val):
    return [v.strip() for v in val.split(",") if v.strip()]


def _normalize_models(models):
    out = []
    for m in models:
        up = m.strip().upper()
        if up in {"VANILLA", "HAR-RV", "HAR_RV"}:
            up = "HAR"
        if up:
            out.append(up)
    # de-dup preserve order
    seen = set()
    final = []
    for m in out:
        if m not in seen:
            final.append(m)
            seen.add(m)
    return final


def _run(cmd, cwd=PROJECT):
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def _run_root(run_name: str) -> str:
    return os.path.join(RUNS_DIR, run_name)


def _ensure_run_dirs(run_root: str):
    os.makedirs(run_root, exist_ok=True)
    for sub in ("predictions", "tables", "figures", "latex"):
        os.makedirs(os.path.join(run_root, sub), exist_ok=True)


def _ensure_scratch_dirs():
    os.makedirs(os.path.join(PROJECT, "run_results", "current_intraday", "predictions"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT, "run_results", "current_intraday", "tables"), exist_ok=True)


def _cleanup_scratch_dirs():
    scratch_root = os.path.join(PROJECT, "run_results", "current_intraday")
    if os.path.isdir(scratch_root):
        shutil.rmtree(scratch_root, ignore_errors=True)


def _guess_next_run_name():
    runs_dir = RUNS_DIR
    if not os.path.isdir(runs_dir):
        return "test_1"
    nums = []
    for name in os.listdir(runs_dir):
        m = re.match(r"test_(\d+)$", name)
        if m:
            nums.append(int(m.group(1)))
    return f"test_{max(nums) + 1}" if nums else "test_1"


def _write_run_meta(run_name: str, meta: dict):
    run_root = _run_root(run_name)
    os.makedirs(run_root, exist_ok=True)
    meta_path = os.path.join(run_root, "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(LATEST_META, "w", encoding="utf-8") as f:
        json.dump({"run": run_name, **meta}, f, indent=2)
    print(f"[ok] run meta saved to {run_root}")


def _print_commands_summary():
    if not os.path.exists(COMMANDS_MD):
        print("[info] result_generation_commands.md not found.")
        return
    print("\nAvailable paper outputs (from result_generation_commands.md):")
    lines = []
    with open(COMMANDS_MD, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    in_post = False
    for ln in lines:
        if "Post-run commands to generate results" in ln:
            in_post = True
            continue
        if not in_post:
            continue
        if ln.strip().startswith("Section"):
            print(f"  {ln.strip()}")
        elif ln.strip().startswith("Figure") or ln.strip().startswith("Fig") or ln.strip().startswith("Table"):
            print(f"    - {ln.strip()}")


def _load_latest_defaults():
    if os.path.exists(LATEST_META):
        try:
            with open(LATEST_META, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "run": data.get("run"),
                "pred_dir": data.get("pred_dir"),
                "tables_dir": data.get("tables_dir"),
                "fig_out": data.get("fig_out"),
                "tex_out": data.get("tex_out"),
                "tickers": data.get("tickers"),
                "models": data.get("models"),
                "local_dir": data.get("local_dir"),
                "n": data.get("n"),
                "warmup": data.get("warmup"),
                "section5_prereq": data.get("section5_prereq"),
            }
        except Exception:
            return {}
    return {}


def _load_run_meta(run_root: str) -> dict:
    meta_path = os.path.join(run_root, "run_meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _abs_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT, path)


def _extra_args(msg: str) -> List[str]:
    raw = input(f"{msg} (leave blank for none): ").strip()
    return shlex.split(raw) if raw else []


def _scratch_pred_dir() -> str:
    return os.path.join(PROJECT, "run_results", "current_intraday", "predictions")


def _scratch_tables_dir() -> str:
    return os.path.join(PROJECT, "run_results", "current_intraday", "tables")


def _sync_predictions_to_scratch(pred_dir: str, tickers: List[str]):
    _ensure_scratch_dirs()
    scratch_pred = _scratch_pred_dir()
    # Clear existing CSVs
    for f in os.listdir(scratch_pred):
        if f.lower().endswith(".csv"):
            os.remove(os.path.join(scratch_pred, f))
    for t in tickers:
        src = os.path.join(pred_dir, f"{t}.csv")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(scratch_pred, f"{t}.csv"))


def _copy_if_exists(src: str, dst: str):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _build_section2_regimes(pred_dir: str, tables_dir: str, tickers: List[str]):
    _sync_predictions_to_scratch(pred_dir, tickers)
    _run([sys.executable, os.path.join("code", "Helpers_for_paper_export_scripts", "Section_2", "run_regime_analysis.py")])
    src = os.path.join(_scratch_tables_dir(), "Section_2_regimes.csv")
    dst = os.path.join(tables_dir, "Section_2_regimes.csv")
    _copy_if_exists(src, dst)
    print(f"[ok] Section 2 regimes table -> {dst}")


def _collect_section4_summaries(tables_dir: str):
    scratch_tables = _scratch_tables_dir()
    if not os.path.isdir(scratch_tables):
        return
    for f in os.listdir(scratch_tables):
        if f.startswith("Section_4_error_advantage_summary_") and f.endswith(".csv"):
            _copy_if_exists(os.path.join(scratch_tables, f), os.path.join(tables_dir, f))


def _combined_dir(pred_dir: str) -> str:
    path = os.path.join(pred_dir, "combined")
    os.makedirs(path, exist_ok=True)
    return path


def _concat_predictions(pred_dir: str, tickers: List[str], out_name: str = "ALL_TICKERS.csv"):
    frames = []
    for t in tickers:
        path = os.path.join(pred_dir, f"{t}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "Ticker" in df.columns:
            df["Ticker"] = t
        else:
            df.insert(0, "Ticker", t)
        frames.append(df)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True, sort=False)
    out_path = os.path.join(_combined_dir(pred_dir), out_name)
    combined.to_csv(out_path, index=False)
    return out_path


def _mean_predictions(pred_dir: str, tickers: List[str], models: List[str], out_name: str = "ALL_MEAN.csv"):
    # inner-join on Date, then mean across tickers for Actual and Predicted_* columns
    joined = None
    for t in tickers:
        path = os.path.join(pred_dir, f"{t}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["Date"])
        cols = ["Date", "Actual"]
        for m in models:
            col = f"Predicted_{m}"
            if col in df.columns:
                cols.append(col)
        df = df[cols].set_index("Date")
        df = df.rename(columns={c: f"{c}_{t}" for c in df.columns})
        joined = df if joined is None else joined.join(df, how="inner")
    if joined is None or joined.empty:
        return None

    def _avg(prefix):
        cols = [c for c in joined.columns if c.startswith(prefix)]
        if not cols:
            return None
        return joined[cols].mean(axis=1)

    actual = _avg("Actual_")
    if actual is None:
        return None

    out = pd.DataFrame({"Actual": actual})
    for m in models:
        pred = _avg(f"Predicted_{m}_")
        if pred is None:
            continue
        out[f"Predicted_{m}"] = pred
        err = out["Actual"] - out[f"Predicted_{m}"]
        out[f"Err_{m}"] = err
        out[f"AbsErr_{m}"] = err.abs()
        denom = (out["Actual"].abs() + out[f"Predicted_{m}"].abs()).clip(lower=1e-12)
        out[f"SMAPE_{m}_pct"] = 200.0 * err.abs() / denom

    out = out.reset_index().rename(columns={"index": "Date"})
    out_path = os.path.join(_combined_dir(pred_dir), out_name)
    out.to_csv(out_path, index=False)
    return out_path


def run_predictions_interactive():
    print("\n=== Predictions Run ===")
    print("Notes:")
    print("- HAR = Vanilla baseline. PM/CP are extensions.")
    print("- Variants: PM_VW, PM_AD, CP_CJ; controls: RAND, CRS.")
    print("- Section 5 needs variants/controls in predictions.")
    run_name = _prompt("Run name", _guess_next_run_name())
    run_root = _run_root(run_name)
    _ensure_run_dirs(run_root)

    pred_dir = os.path.join(run_root, "predictions")
    tables_dir = os.path.join(run_root, "tables")
    fig_out = os.path.join(run_root, "figures")
    tex_out = os.path.join(run_root, "latex")

    local_dir = _prompt("Local clean data dir", "data/market_data/clean")
    tickers = _prompt("Tickers (comma-separated)", DEFAULT_TICKERS_10)
    models_in = _prompt("Models (comma-separated, Vanilla=HAR)", DEFAULT_MODELS)
    models = _normalize_models(_split_csv(models_in))

    want_section5 = _prompt_bool("Include Section 5 prerequisites (PM_VW, PM_AD, CP_CJ, RAND, CRS)?", False)
    if want_section5:
        needed = ["PM_VW", "PM_AD", "CP_CJ", "RAND", "CRS"]
        for m in needed:
            if m not in models:
                models.append(m)
        print(f"[info] Added Section 5 prerequisites: {','.join(needed)}")

    n = int(_prompt("Lookback n", "22"))
    warmup = int(_prompt("Warmup", "600"))
    run_intraday = _prompt_bool("Run intraday benchmark?", True)
    run_daily = _prompt_bool("Run daily benchmark?", False)
    auto_regimes = _prompt_bool("Auto-generate Section 2 regimes table after intraday?", True)

    if run_intraday:
        cmd = [
            sys.executable,
            os.path.join("code", "intraday_benchmark.py"),
            "--tickers", tickers,
            "--models", ",".join(models),
            "--local-dir", local_dir,
            "--outdir", run_root,
            "--n", str(n),
            "--warmup", str(warmup),
            "--require-explicit",
        ]
        if not want_section5:
            cmd.append("--skip-section5")
        _run(cmd)

        ticker_list = _split_csv(tickers)
        concat_path = _concat_predictions(pred_dir, ticker_list)
        if concat_path:
            print(f"[ok] combined tickers file: {concat_path}")
        if len(ticker_list) > 1:
            mean_path = _mean_predictions(pred_dir, ticker_list, models)
            if mean_path:
                print(f"[ok] combined mean file: {mean_path}")

        if auto_regimes:
            _build_section2_regimes(pred_dir, tables_dir, ticker_list)

    if run_daily:
        daily_tmp = os.path.join(run_root, "_daily_tmp")
        cmd = [
            sys.executable,
            os.path.join("code", "daily_benchmark.py"),
            "--tickers", tickers,
            "--models", ",".join(models),
            "--local-dir", local_dir,
            "--outdir", daily_tmp,
            "--n", str(n),
            "--warmup", str(max(30, int(warmup / 10))),
            "--require-explicit",
        ]
        _run(cmd)
        daily_pred_src = os.path.join(daily_tmp, "predictions")
        daily_tbl_src = os.path.join(daily_tmp, "tables")
        daily_pred_dst = os.path.join(pred_dir, "daily")
        daily_tbl_dst = os.path.join(tables_dir, "daily")
        os.makedirs(daily_pred_dst, exist_ok=True)
        os.makedirs(daily_tbl_dst, exist_ok=True)
        if os.path.isdir(daily_pred_src):
            for f in os.listdir(daily_pred_src):
                shutil.copy2(os.path.join(daily_pred_src, f), os.path.join(daily_pred_dst, f))
        if os.path.isdir(daily_tbl_src):
            for f in os.listdir(daily_tbl_src):
                shutil.copy2(os.path.join(daily_tbl_src, f), os.path.join(daily_tbl_dst, f))
        shutil.rmtree(daily_tmp, ignore_errors=True)

    meta = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "tickers": tickers,
        "models": ",".join(models),
        "local_dir": local_dir,
        "n": n,
        "warmup": warmup,
        "section5_prereq": bool(want_section5),
        "intraday": run_intraday,
        "daily": run_daily,
        "pred_dir": pred_dir,
        "tables_dir": tables_dir,
        "fig_out": fig_out,
        "tex_out": tex_out,
    }
    _cleanup_scratch_dirs()
    _write_run_meta(run_name, meta)


def run_outputs_interactive():
    print("\n=== Paper Outputs ===")
    _print_commands_summary()
    print("Note: Section 2 + Section 4 helpers use the current generated-output scratch area under run_results/current_intraday.")

    sections = _prompt("Sections to run (e.g., 1,2,3,4,6 or all)", "1,2,3,4,6")
    if sections.strip().lower() == "all":
        chosen = {"1", "2", "3", "4", "5", "6"}
    else:
        chosen = set(_split_csv(sections))

    latest = _load_latest_defaults()
    default_run = latest.get("run") or _guess_next_run_name()
    run_name = _prompt("Run name", default_run)
    run_root = _run_root(run_name)
    _ensure_run_dirs(run_root)
    run_meta = _load_run_meta(run_root)

    pred_dir = _prompt("Predictions dir", run_meta.get("pred_dir") or os.path.join(run_root, "predictions"))
    tables_dir = _prompt("Tables dir", run_meta.get("tables_dir") or os.path.join(run_root, "tables"))
    fig_out = _prompt("Figures outdir", run_meta.get("fig_out") or os.path.join(run_root, "figures"))
    tex_out = _prompt("LaTeX outdir", run_meta.get("tex_out") or os.path.join(run_root, "latex"))
    models = _normalize_models(_split_csv(_prompt("Models for plots (HAR,PM,CP...)", run_meta.get("models") or DEFAULT_MODELS)))
    baseline = _prompt("Baseline model", "HAR")
    tickers = _prompt("Tickers (comma-separated)", run_meta.get("tickers") or DEFAULT_TICKERS_10)
    section5_prereq = bool(run_meta.get("section5_prereq", False))

    print("\nSelection mode:")
    print("1) Default (recommended)")
    print("2) Custom (choose variants per section)")
    mode = _prompt("Choose", "1")
    custom = (mode == "2")

    if "1" in chosen:
        print("\n[Section 1] Overall performance table (mean across assets).")
        if not custom or _prompt_bool("Generate Table 1a (LaTeX)?", True):
            cmd = [
                sys.executable,
                os.path.join("code", "paper_export", "Section_1", "latex_table", "Section_1_table_1_overall.py"),
                "--csv", os.path.join(tables_dir, "Section_1_table_1a_overall.csv"),
                "--out", os.path.join(tex_out, "Section_1_table_1a_overall.tex"),
                "--context", "intraday",
            ]
            _run(cmd)

    if "2" in chosen:
        print("\n[Section 2] Regime table + figures (low/med/high volatility regimes).")
        _build_section2_regimes(pred_dir, tables_dir, _split_csv(tickers))
        if not custom or _prompt_bool("Generate Table 2 (LaTeX)?", True):
            include_variants = True if not custom else _prompt_bool("Include variants in Table 2?", True)
            cmd = [
                sys.executable,
                os.path.join("code", "paper_export", "Section_2", "latex_table", "Section_2_table_2_regimes.py"),
                "--csv", os.path.join(tables_dir, "Section_2_regimes.csv"),
                "--out", os.path.join(tex_out, "Section_2_table_2_regimes.tex"),
            ]
            if include_variants:
                cmd.append("--include-variants")
            _run(cmd)
        if not custom or _prompt_bool("Generate Section 2 figures?", True):
            rel = True if not custom else _prompt_bool("Relative improvement figure?", True)
            absf = True if not custom else _prompt_bool("Absolute SMAPE figure?", True)
            rel_extra = _extra_args("Extra args for relative figure") if custom and rel else []
            abs_extra = _extra_args("Extra args for absolute figure") if custom and absf else []
            if rel:
                cmd = [
                    sys.executable,
                    os.path.join("code", "paper_export", "Section_2", "figure", "Section_2_fig_regimes.py"),
                    "--csv", os.path.join(tables_dir, "Section_2_regimes.csv"),
                    "--outdir", fig_out,
                ] + rel_extra
                _run(cmd)
            if absf:
                cmd = [
                    sys.executable,
                    os.path.join("code", "paper_export", "Section_2", "figure", "Section_2_fig_regimes.py"),
                    "--csv", os.path.join(tables_dir, "Section_2_regimes.csv"),
                    "--outdir", fig_out,
                    "--absolute",
                ] + abs_extra
                _run(cmd)

    if "3" in chosen:
        print("\n[Section 3] Temporal stability plots (rolling performance over time).")
        if not custom or _prompt_bool("Generate Section 3 figures?", True):
            print("  - Single ticker: plots that asset only.")
            print("  - All tickers: loops through each ticker.")
            print("  - Combined mean: averages all tickers into ALL_MEAN.")
            print("Asset mode: 1) single ticker  2) all tickers  3) combined mean (ALL_MEAN)")
            asset_mode = _prompt("Choose asset mode", "1") if custom else "1"
            assets = []
            tick_list = _split_csv(tickers)
            if asset_mode == "2":
                assets = tick_list
            elif asset_mode == "3":
                _mean_predictions(_abs_path(pred_dir), tick_list, models, out_name="ALL_MEAN.csv")
                assets = ["ALL_MEAN"]
            else:
                assets = [_prompt("Asset ticker", "SPY") if custom else "SPY"]

            mono = True if not custom else _prompt_bool("Monochrome figure?", True)
            color = True if not custom else _prompt_bool("Color figure?", True)
            mono_extra = _extra_args("Extra args for monochrome") if custom and mono else []
            color_extra = _extra_args("Extra args for color") if custom and color else []

            for a in assets:
                use_pred_dir = pred_dir
                if a == "ALL_MEAN":
                    use_pred_dir = os.path.join(pred_dir, "combined")
                if mono:
                    cmd = [
                        sys.executable,
                        os.path.join("code", "paper_export", "Section_3", "figure", "Section_3_fig_2_temporal_stability.py"),
                        "--pred-dir", use_pred_dir,
                        "--asset", a,
                        "--models", ",".join(models),
                        "--window", "78",
                        "--smooth", "ema",
                        "--smooth-span", "39",
                        "--resample", "W",
                        "--trend",
                        "--scale", "90,170",
                        "--style", "monochrome",
                        "--outdir", fig_out,
                    ] + mono_extra
                    _run(cmd)
                if color:
                    cmd = [
                        sys.executable,
                        os.path.join("code", "paper_export", "Section_3", "figure", "Section_3_fig_2_temporal_stability.py"),
                        "--pred-dir", use_pred_dir,
                        "--asset", a,
                        "--models", ",".join(models),
                        "--window", "78",
                        "--smooth", "ma",
                        "--smooth-span", "13",
                        "--resample", "W",
                        "--style", "color",
                        "--outdir", fig_out,
                    ] + color_extra
                    _run(cmd)

    if "4" in chosen:
        comp_models = [m for m in models if m != baseline]
        if not comp_models:
            print("[warn] Section 4 needs at least one model besides baseline; skipping.")
        else:
            print("\n[Section 4] Error advantage vs baseline over time.")
            if not custom or _prompt_bool("Generate Section 4 figures?", True):
                print("  - Single ticker: plots that asset only.")
                print("  - All tickers: loops through each ticker.")
                print("  - Combined mean: averages all tickers into ALL_MEAN.")
                print("Asset mode: 1) single ticker  2) all tickers  3) combined mean (ALL_MEAN)")
                asset_mode = _prompt("Choose asset mode", "1") if custom else "1"
                assets = []
                tick_list = _split_csv(tickers)
                if asset_mode == "2":
                    assets = tick_list
                elif asset_mode == "3":
                    _mean_predictions(_abs_path(pred_dir), tick_list, models, out_name="ALL_MEAN.csv")
                    assets = ["ALL_MEAN"]
                else:
                    assets = [_prompt("Asset ticker", "SPY") if custom else "SPY"]

                smooth = True if not custom else _prompt_bool("Smoothed/resampled figure?", True)
                raw = True if not custom else _prompt_bool("Raw (no resample) figure?", True)
                smooth_extra = _extra_args("Extra args for smoothed figure") if custom and smooth else []
                raw_extra = _extra_args("Extra args for raw figure") if custom and raw else []
                for a in assets:
                    _ensure_scratch_dirs()
                    use_pred_dir = pred_dir
                    if a == "ALL_MEAN":
                        use_pred_dir = os.path.join(pred_dir, "combined")
                    if smooth:
                        cmd = [
                            sys.executable,
                            os.path.join("code", "paper_export", "Section_4", "render_and_figure", "Section_4_render_and_fig_error_advantage.py"),
                            "--pred-dir", use_pred_dir,
                            "--asset", a,
                            "--baseline", baseline,
                            "--models", ",".join(comp_models),
                            "--metric", "smape",
                            "--resample", "W",
                            "--smooth", "ema",
                            "--smooth-span", "7",
                            "--clip-pctl", "1,99",
                            "--scale=-50,50",
                            "--outdir", fig_out,
                        ] + smooth_extra
                        _run(cmd)
                    if raw:
                        cmd = [
                            sys.executable,
                            os.path.join("code", "paper_export", "Section_4", "render_and_figure", "Section_4_render_and_fig_error_advantage.py"),
                            "--pred-dir", use_pred_dir,
                            "--asset", a,
                            "--baseline", baseline,
                            "--models", ",".join(comp_models),
                            "--metric", "smape",
                            "--scale=-50,50",
                            "--outdir", fig_out,
                        ] + raw_extra
                        _run(cmd)
                _collect_section4_summaries(tables_dir)

    if "5" in chosen:
        print("[note] Section 5 requires PM/CP variants and random controls to be present in predictions.")
        if not section5_prereq:
            if _prompt_bool("Prerequisites not present. Re-run intraday with Section 5 prerequisites now?", False):
                meta_local = run_meta.get("local_dir") or _prompt("Local clean data dir", "data/market_data/clean")
                meta_n = run_meta.get("n") or int(_prompt("Lookback n", "22"))
                meta_warmup = run_meta.get("warmup") or int(_prompt("Warmup", "600"))
                needed = ["PM_VW", "PM_AD", "CP_CJ", "RAND", "CRS"]
                models_ext = models[:]
                for m in needed:
                    if m not in models_ext:
                        models_ext.append(m)
                cmd = [
                    sys.executable,
                    os.path.join("code", "intraday_benchmark.py"),
                    "--tickers", tickers,
                    "--models", ",".join(models_ext),
                    "--local-dir", meta_local,
                    "--outdir", run_root,
                    "--n", str(meta_n),
                    "--warmup", str(meta_warmup),
                    "--require-explicit",
                ]
                _run(cmd)
                # refresh combined predictions
                tick_list = _split_csv(tickers)
                _concat_predictions(pred_dir, tick_list)
                if len(tick_list) > 1:
                    _mean_predictions(pred_dir, tick_list, models_ext)
                section5_prereq = True
                run_meta.update({
                    "models": ",".join(models_ext),
                    "local_dir": meta_local,
                    "n": meta_n,
                    "warmup": meta_warmup,
                    "section5_prereq": True,
                })
                _write_run_meta(run_name, {
                    "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "tickers": tickers,
                    "models": ",".join(models_ext),
                    "local_dir": meta_local,
                    "n": meta_n,
                    "warmup": meta_warmup,
                    "section5_prereq": True,
                    "intraday": True,
                    "daily": run_meta.get("daily", False),
                    "pred_dir": pred_dir,
                    "tables_dir": tables_dir,
                    "fig_out": fig_out,
                    "tex_out": tex_out,
                })
        if _prompt_bool("Run Section 5 helper + figures/tables?", section5_prereq):
            assets_val = tickers if not custom else _prompt("Assets (comma-separated)", tickers)
            _run([
                sys.executable,
                os.path.join("code", "Helpers_for_paper_export_scripts", "Section_5", "sweep_pm_capacity.py"),
                "--assets", assets_val,
                "--k-grid", "3,4,5,6,7,8",
                "--n", "390",
                "--warmup", "200",
                "--local-dir", "data/market_data/clean",
                "--out-csv", os.path.join(tables_dir, "Section_5_fig_4_capacity_curve.csv"),
            ])
            _run([
                sys.executable,
                os.path.join("code", "paper_export", "Section_5", "figure", "Section_5_fig_4_capacity_curve.py"),
                "--csv", os.path.join(tables_dir, "Section_5_fig_4_capacity_curve.csv"),
                "--outdir", fig_out,
            ])
            _run([
                sys.executable,
                os.path.join("code", "paper_export", "Section_5", "figure", "Section_5_fig_5_pm_modifiers_efficacy.py"),
                "--tables-dir", tables_dir,
                "--outdir", fig_out,
                "--style", "mono",
            ])
            _run([
                sys.executable,
                os.path.join("code", "paper_export", "Section_5", "figure", "Section_5_fig_6_cp_vs_cj_variant.py"),
                "--tables-dir", tables_dir,
                "--outdir", fig_out,
                "--style", "mono",
                "--metrics", "smape,mae,rmse,diracc",
            ])
            _run([
                sys.executable,
                os.path.join("code", "paper_export", "Section_5", "latex_table", "Section_5_table_3_param_efficiency.py"),
                "--csv", os.path.join(tables_dir, "Section_5_table_3_param_eff_joined.csv"),
                "--out", os.path.join(tex_out, "Section_5_table_3_param_eff.tex"),
            ])
            _run([
                sys.executable,
                os.path.join("code", "paper_export", "Section_5", "latex_table", "Section_5_table_4_random_controls.py"),
                "--in", os.path.join(tables_dir, "Section_5_table_4_random_controls.csv"),
                "--outdir", tex_out,
                "--scope", "intraday",
            ])

    if "6" in chosen:
        comp_models = [m for m in models if m != baseline]
        if not comp_models:
            print("[warn] Section 6 needs at least one model besides baseline; skipping.")
        else:
            print("\n[Section 6] Group-level significance by asset category.")
            gen_table = True if not custom else _prompt_bool("Generate Section 6 table?", True)
            gen_fig = True if not custom else _prompt_bool("Generate Section 6 figure?", True)
            if gen_table or gen_fig:
                print("[note] Section 6 script outputs both table and figure in one run.")
                cmd = [
                    sys.executable,
                    os.path.join("code", "paper_export", "Section_6", "figure_and_latex_table", "Section_6_fig_7_and_table_5_asset_groups.py"),
                    "--pred-dir", pred_dir,
                    "--tables-dir", tables_dir,
                    "--out-tex", os.path.join(tex_out, "Section_6_table_5_asset_groups.tex"),
                    "--out-fig", os.path.join(fig_out, "Section_6_fig_7_asset_groups.pdf"),
                    "--baseline", baseline,
                    "--models", ",".join(comp_models),
                    "--tickers", tickers,
                    "--auto-build",
                ]
                _run(cmd)

    meta = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "pred_dir": pred_dir,
        "tables_dir": tables_dir,
        "fig_out": fig_out,
        "tex_out": tex_out,
        "sections": sorted(chosen),
        "tickers": tickers,
        "models": ",".join(models),
        "section5_prereq": section5_prereq,
    }
    _write_run_meta(run_name, meta)
    _cleanup_scratch_dirs()


def main():
    print("Interactive Pipeline")
    print("1) Run predictions")
    print("2) Generate paper outputs")
    print("3) Both")
    print("4) Exit")
    choice = _prompt("Choose", "1")
    if choice == "1":
        run_predictions_interactive()
    elif choice == "2":
        run_outputs_interactive()
    elif choice == "3":
        run_predictions_interactive()
        run_outputs_interactive()
    else:
        print("Bye.")


if __name__ == "__main__":
    main()
