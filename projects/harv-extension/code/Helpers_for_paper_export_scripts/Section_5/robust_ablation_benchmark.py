import argparse
import gc
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils.data_utils import (  # noqa: E402
    calculate_intraday_realized_volatility,
    fetch_intraday_data,
    fit_and_predict_extended,
)
from utils.harvey_utils import add_harv_tcj_terms  # noqa: E402
from utils.models_utils import (  # noqa: E402
    add_prime_modulo_terms,
    build_minimal_primes,
    contig_prime_modulo,
)


BASE_FEATURES = ["RV_d", "RV_w", "RV_m"]
STOP_REQUESTED = False


@dataclass
class ModelSpec:
    name: str
    suite: str
    family: str
    feature_count: int | None
    builder: Callable
    direct_predictor: Callable | None = None


def _handle_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[STOP] received signal {signum}; will stop after current checkpoint", flush=True)


signal.signal(signal.SIGTERM, _handle_stop)
if hasattr(signal, "SIGINT"):
    signal.signal(signal.SIGINT, _handle_stop)


def _now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_s(sec):
    if sec >= 3600:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        return f"{h}h{m:02d}m"
    if sec >= 60:
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m}m{s:02d}s"
    return f"{sec:.1f}s"


def _append_line(path, line):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def _feature_cols(frame, prefixes):
    return [c for c in frame.columns if any(c.startswith(prefix) for prefix in prefixes)]


def _base_plus(extras):
    return BASE_FEATURES + list(extras)


def _lag_average_features(frame, groups, prefix):
    out = frame.copy()
    for idx, lags in enumerate(groups, start=1):
        clean_lags = [int(lag) for lag in lags if int(lag) >= 1]
        if not clean_lags:
            continue
        lagged = pd.concat([out["RV_d"].shift(lag) for lag in clean_lags], axis=1)
        out[f"{prefix}_{idx:02d}"] = lagged.mean(axis=1)
    return out


def _equal_window_builder(k):
    def build(data, n):
        lags = np.arange(1, int(n) + 1)
        groups = np.array_split(lags, int(k))
        frame = _lag_average_features(data, groups, f"EQWIN_FC{k}")
        extras = _feature_cols(frame, [f"EQWIN_FC{k}_"])
        return frame, _base_plus(extras), len(extras)

    return build


def _prefix_window_builder(k):
    def build(data, n):
        lengths = np.unique(np.rint(np.linspace(1, int(n), int(k))).astype(int))
        groups = [np.arange(1, length + 1) for length in lengths]
        frame = _lag_average_features(data, groups, f"PREFIX_FC{k}")
        extras = _feature_cols(frame, [f"PREFIX_FC{k}_"])
        return frame, _base_plus(extras), len(extras)

    return build


def _shifted_window_builder(k, offset):
    def build(data, n):
        lags = list(range(1, int(n) + 1))
        shift = int(offset) % len(lags)
        rotated = lags[shift:] + lags[:shift]
        groups = np.array_split(np.array(rotated), int(k))
        frame = _lag_average_features(data, groups, f"SHIFTWIN_FC{k}_O{offset}")
        extras = _feature_cols(frame, [f"SHIFTWIN_FC{k}_O{offset}_"])
        return frame, _base_plus(extras), len(extras)

    return build


def _pm_builder(k=None):
    def build(data, n):
        frame = add_prime_modulo_terms(data.copy(), n)
        extras = _feature_cols(frame, ["PM_"])
        if k is not None:
            extras = extras[: min(int(k), len(extras))]
        return frame, _base_plus(extras), len(extras)

    return build


def _cp_builder(k=None):
    def build(data, n):
        frame = contig_prime_modulo(data.copy(), n)
        extras = _feature_cols(frame, ["CP_"])
        if k is not None:
            extras = extras[: min(int(k), len(extras))]
        return frame, _base_plus(extras), len(extras)

    return build


def _modulo_builder(moduli, k, label):
    def build(data, n):
        frame = data.copy()
        lags = list(range(1, int(n) + 1))
        created = []
        for modulus in moduli:
            for remainder in range(modulus):
                relevant = [lag for lag in lags if lag % modulus == remainder]
                if not relevant:
                    continue
                lagged = pd.concat([frame["RV_d"].shift(lag) for lag in relevant], axis=1)
                col = f"{label}_m{modulus}_r{remainder}"
                frame[col] = lagged.mean(axis=1)
                created.append(col)
        extras = created[: min(int(k), len(created))]
        return frame, _base_plus(extras), len(extras)

    return build


def _random_subset_builder(k, seed):
    def build(data, n):
        frame = data.copy()
        rng = np.random.default_rng(int(seed))
        created = []
        lags = np.arange(1, int(n) + 1)
        for idx in range(1, int(k) + 1):
            size = int(rng.integers(1, len(lags) + 1))
            selected = np.sort(rng.choice(lags, size=size, replace=False))
            lagged = pd.concat([frame["RV_d"].shift(int(lag)) for lag in selected], axis=1)
            col = f"RAND_FC{k}_S{seed}_{idx:02d}"
            frame[col] = lagged.mean(axis=1)
            created.append(col)
        return frame, _base_plus(created), len(created)

    return build


def _contiguous_random_builder(k, seed):
    def build(data, n):
        frame = data.copy()
        rng = np.random.default_rng(int(seed))
        created = []
        horizon = int(n)
        for idx in range(1, int(k) + 1):
            length = int(rng.integers(1, horizon + 1))
            start = int(rng.integers(1, horizon - length + 2))
            selected = np.arange(start, start + length)
            lagged = pd.concat([frame["RV_d"].shift(int(lag)) for lag in selected], axis=1)
            col = f"CRS_FC{k}_S{seed}_{idx:02d}"
            frame[col] = lagged.mean(axis=1)
            created.append(col)
        return frame, _base_plus(created), len(created)

    return build


def _har_builder(data, n):
    return data.copy(), BASE_FEATURES, 0


def _harq_builder(data, n):
    frame = data.copy()
    frame["RQ_d"] = frame["SR"] ** 2
    frame["RQ_w"] = frame["RQ_d"].rolling(window=78, min_periods=78).mean()
    frame["RQ_m"] = frame["RQ_d"].rolling(window=78 * 21, min_periods=78 * 21).mean()
    frame["RVd_x_RQd"] = frame["RV_d"] * frame["RQ_d"]
    extras = ["RQ_d", "RQ_w", "RQ_m", "RVd_x_RQd"]
    return frame, _base_plus(extras), len(extras)


def _har_tcj_builder(data, n):
    frame = add_harv_tcj_terms(data.copy(), n)
    extras = [c for c in frame.columns if c.startswith("RV_TC")]
    return frame, _base_plus(extras), len(extras)


def _ewma_predict(data, n, warmup, model_name, lam=0.94):
    sr_ewma = data["SR"].ewm(alpha=(1.0 - lam), adjust=False).mean()
    rows = []
    for i in range(int(n) + int(warmup), len(data) - 1):
        pred = float(np.sqrt(max(sr_ewma.iloc[i], 0.0)))
        actual = float(data.iloc[i + 1]["RV_d"])
        err = actual - pred
        denom = max(1e-12, abs(actual) + abs(pred))
        rows.append(
            {
                "Date": data.index[i + 1],
                "Actual": actual,
                f"Predicted_{model_name}": pred,
                f"Err_{model_name}": err,
                f"AbsErr_{model_name}": abs(err),
                f"SMAPE_{model_name}_pct": 200.0 * abs(err) / denom,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.set_index("Date")


def _garch11_predict(data, n, warmup, model_name, refit_every=390):
    try:
        from arch import arch_model
    except ImportError as exc:
        raise RuntimeError("GARCH11 requires the 'arch' package") from exc

    returns = data["Log_Return"].replace([np.inf, -np.inf], np.nan)
    rows = []
    omega = alpha = beta = sigma2_current = None
    last_fit_i = None

    for i in range(int(n) + int(warmup), len(data) - 1):
        if STOP_REQUESTED:
            break

        need_fit = omega is None or last_fit_i is None or (i - last_fit_i) >= int(refit_every)
        if need_fit:
            train = (returns.iloc[: i + 1].dropna() * 100.0).astype(float)
            if len(train) < 100:
                continue
            model = arch_model(
                train,
                mean="Zero",
                vol="GARCH",
                p=1,
                q=1,
                dist="normal",
                rescale=False,
            )
            res = model.fit(disp="off", show_warning=False, options={"maxiter": 80})
            params = res.params
            omega = float(params.get("omega", np.nan))
            alpha = float(params.get("alpha[1]", np.nan))
            beta = float(params.get("beta[1]", np.nan))
            sigma2_current = float(res.conditional_volatility.iloc[-1] ** 2)
            last_fit_i = i
            if not np.isfinite([omega, alpha, beta, sigma2_current]).all():
                omega = alpha = beta = sigma2_current = None
                continue

        r_i = returns.iloc[i]
        if not np.isfinite(r_i) or sigma2_current is None:
            continue
        r_scaled = float(r_i * 100.0)
        sigma2_next = max(omega + alpha * (r_scaled ** 2) + beta * sigma2_current, 0.0)
        pred = float(np.sqrt(sigma2_next) / 100.0)
        actual = float(data.iloc[i + 1]["RV_d"])
        err = actual - pred
        denom = max(1e-12, abs(actual) + abs(pred))
        rows.append(
            {
                "Date": data.index[i + 1],
                "Actual": actual,
                f"Predicted_{model_name}": pred,
                f"Err_{model_name}": err,
                f"AbsErr_{model_name}": abs(err),
                f"SMAPE_{model_name}_pct": 200.0 * abs(err) / denom,
            }
        )
        sigma2_current = sigma2_next

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.set_index("Date")


def build_specs(suite, k_grid, seeds):
    k_values = [int(k) for k in k_grid]
    seed_values = [int(s) for s in seeds]
    specs = []

    if suite in {"all", "core"}:
        specs.extend(
            [
                ModelSpec("HAR", "core", "HAR", 0, _har_builder),
                ModelSpec("PM", "core", "PM", None, _pm_builder(None)),
                ModelSpec("CP", "core", "CP", None, _cp_builder(None)),
            ]
        )

    if suite in {"all", "capacity"}:
        for k in k_values:
            specs.append(ModelSpec(f"PM_FC{k}", "capacity", "PM_FC", k, _pm_builder(k)))
            specs.append(ModelSpec(f"CP_FC{k}", "capacity", "CP_FC", k, _cp_builder(k)))
            specs.append(ModelSpec(f"EQWIN_FC{k}", "capacity", "EQWIN", k, _equal_window_builder(k)))
            specs.append(ModelSpec(f"PREFIX_FC{k}", "capacity", "PREFIX", k, _prefix_window_builder(k)))

    if suite in {"all", "contiguous"}:
        for k in [6, 9]:
            for offset in [1, 3, 5]:
                specs.append(
                    ModelSpec(
                        f"SHIFTWIN_FC{k}_O{offset}",
                        "contiguous",
                        "SHIFTWIN",
                        k,
                        _shifted_window_builder(k, offset),
                    )
                )

    if suite in {"all", "modulo"}:
        for k in [6, 9]:
            specs.append(
                ModelSpec(
                    f"COMP_COPRIME_FC{k}",
                    "modulo",
                    "COMP_COPRIME",
                    k,
                    _modulo_builder([4, 9], k, f"COMP_COPRIME_FC{k}"),
                )
            )
            specs.append(
                ModelSpec(
                    f"COMP_NONCOPRIME_FC{k}",
                    "modulo",
                    "COMP_NONCOPRIME",
                    k,
                    _modulo_builder([4, 6], k, f"COMP_NONCOPRIME_FC{k}"),
                )
            )

    if suite in {"all", "random"}:
        for k in [3, 6, 9]:
            for seed in seed_values:
                specs.append(
                    ModelSpec(
                        f"RAND_FC{k}_S{seed}",
                        "random",
                        "RAND",
                        k,
                        _random_subset_builder(k, seed),
                    )
                )
                specs.append(
                    ModelSpec(
                        f"CRS_FC{k}_S{seed}",
                        "random",
                        "CRS",
                        k,
                        _contiguous_random_builder(k, seed),
                    )
                )

    if suite in {"all", "external"}:
        specs.extend(
            [
                ModelSpec("EWMA", "external", "EWMA", 0, _har_builder, direct_predictor=_ewma_predict),
                ModelSpec("HARQ", "external", "HARQ", 4, _harq_builder),
                ModelSpec("HAR_TCJ", "external", "HAR_TCJ", None, _har_tcj_builder),
                ModelSpec("GARCH11", "external", "GARCH11", 0, _har_builder, direct_predictor=_garch11_predict),
            ]
        )

    return specs


def _completed_models_from_checkpoint(out_csv, specs):
    if not os.path.exists(out_csv):
        return set()
    try:
        header = pd.read_csv(out_csv, nrows=0).columns
    except Exception:
        return set()
    wanted = {spec.name for spec in specs}
    return {col.replace("Predicted_", "") for col in header if col.startswith("Predicted_") and col.replace("Predicted_", "") in wanted}


def _append_model_checkpoint(out_csv, frame, model_name):
    new = frame.copy()
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if os.path.exists(out_csv):
        acc = pd.read_csv(out_csv, parse_dates=["Date"]).set_index("Date")
        new_actual = new["Actual"].copy() if "Actual" in new.columns else None
        if "Actual" in acc.columns and "Actual" in new.columns:
            new = new.drop(columns=["Actual"])
        drop_cols = [c for c in new.columns if c in acc.columns]
        if drop_cols:
            acc = acc.drop(columns=drop_cols)
        joined = acc.join(new, how="outer")
        if new_actual is not None and "Actual" in joined.columns:
            joined["Actual"] = joined["Actual"].combine_first(new_actual.reindex(joined.index))
    else:
        joined = new
    joined.reset_index().rename(columns={"index": "Date"}).to_csv(out_csv, index=False)
    print(f"[CHECKPOINT] {model_name} -> {out_csv}", flush=True)


def _write_status(status_csv, row):
    os.makedirs(os.path.dirname(status_csv), exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(status_csv, mode="a", index=False, header=not os.path.exists(status_csv))


def _load_volatility(ticker, local_dir):
    raw = fetch_intraday_data(ticker, use_local=True, local_dir=local_dir)
    vol = calculate_intraday_realized_volatility(raw)
    vol["Log_Return"] = raw["Log_Return"].reindex(vol.index)
    return vol.replace([np.inf, -np.inf], np.nan)


def run_ticker_suite(ticker, suite, specs, args):
    t0 = time.perf_counter()
    pred_dir = Path(args.outdir) / "predictions"
    status_dir = Path(args.outdir) / "status"
    log_dir = Path(args.outdir) / "logs"
    pred_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    out_csv = pred_dir / f"{ticker}_{suite}.csv"
    status_csv = status_dir / f"{ticker}_{suite}_status.csv"
    log_path = log_dir / f"{ticker}_{suite}_progress.log"

    completed = _completed_models_from_checkpoint(out_csv, specs) if args.resume else set()
    if completed:
        print(f"[RESUME] {ticker}/{suite}: {', '.join(sorted(completed))}", flush=True)

    _append_line(str(log_path), f"[{_now()}] START ticker={ticker} suite={suite}")
    print(f"[LOAD] {ticker} local data", flush=True)
    data = _load_volatility(ticker, args.local_dir)
    print(f"[LOAD] {ticker} rows={len(data)}", flush=True)

    for pos, spec in enumerate(specs, start=1):
        if args.max_models and pos > args.max_models:
            print(f"[LIMIT] stopping after {args.max_models} models", flush=True)
            break
        if STOP_REQUESTED:
            _append_line(str(log_path), f"[{_now()}] STOP requested before {spec.name}")
            break
        if spec.name in completed:
            print(f"[RESUME] {ticker}/{suite}/{spec.name} already complete", flush=True)
            continue

        started = time.perf_counter()
        print(f"[MODEL] {ticker}/{suite}/{spec.name} start ({pos}/{len(specs)})", flush=True)
        try:
            if spec.direct_predictor is not None:
                preds = spec.direct_predictor(data.copy(), args.n, args.warmup, spec.name)
                actual_feature_count = spec.feature_count
            else:
                extended, features, actual_feature_count = spec.builder(data.copy(), args.n)
                features = [c for c in features if c in extended.columns]
                preds = fit_and_predict_extended(extended, features, args.n, args.warmup, model_name=spec.name)

            if preds is None or preds.empty:
                raise RuntimeError("model produced no predictions")

            keep = [
                c
                for c in [
                    "Actual",
                    f"Predicted_{spec.name}",
                    f"Err_{spec.name}",
                    f"AbsErr_{spec.name}",
                    f"SMAPE_{spec.name}_pct",
                ]
                if c in preds.columns
            ]
            _append_model_checkpoint(str(out_csv), preds[keep], spec.name)
            seconds = time.perf_counter() - started
            _write_status(
                str(status_csv),
                {
                    "time_utc": _now(),
                    "ticker": ticker,
                    "suite": suite,
                    "model": spec.name,
                    "family": spec.family,
                    "feature_count": actual_feature_count,
                    "status": "complete",
                    "n_obs": len(preds),
                    "seconds": seconds,
                    "error": "",
                },
            )
            _append_line(str(log_path), f"[{_now()}] COMPLETE {spec.name} rows={len(preds)} seconds={seconds:.3f}")
            print(f"[MODEL] {ticker}/{suite}/{spec.name} done in {_fmt_s(seconds)} rows={len(preds)}", flush=True)

        except Exception as exc:
            seconds = time.perf_counter() - started
            err = f"{type(exc).__name__}: {exc}"
            _write_status(
                str(status_csv),
                {
                    "time_utc": _now(),
                    "ticker": ticker,
                    "suite": suite,
                    "model": spec.name,
                    "family": spec.family,
                    "feature_count": spec.feature_count,
                    "status": "failed",
                    "n_obs": 0,
                    "seconds": seconds,
                    "error": err,
                },
            )
            _append_line(str(log_path), f"[{_now()}] FAILED {spec.name} after {seconds:.3f}s: {err}")
            tb_path = log_dir / f"{ticker}_{suite}_{spec.name}_traceback.txt"
            with open(tb_path, "w", encoding="utf-8") as f:
                f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            print(f"[MODEL] {ticker}/{suite}/{spec.name} failed: {err}", flush=True)

        finally:
            gc.collect()

    elapsed = time.perf_counter() - t0
    _append_line(str(log_path), f"[{_now()}] END ticker={ticker} suite={suite} seconds={elapsed:.3f}")
    print(f"[DONE] {ticker}/{suite} total {_fmt_s(elapsed)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Robust Section 5 ablation benchmark")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--suite", required=True, choices=["core", "capacity", "contiguous", "modulo", "random", "external", "all"])
    parser.add_argument("--local-dir", default="code/Datasets/clean")
    parser.add_argument("--outdir", default="code/paper/runs/section5_robust")
    parser.add_argument("--n", type=int, default=22)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--k-grid", default="3,6,9")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--max-models", type=int, default=0)
    args = parser.parse_args()

    k_grid = [int(x.strip()) for x in args.k_grid.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    specs = build_specs(args.suite, k_grid=k_grid, seeds=seeds)

    if args.list_models:
        for spec in specs:
            print(f"{spec.name},{spec.suite},{spec.family},{spec.feature_count}")
        return

    run_ticker_suite(args.ticker.upper(), args.suite, specs, args)


if __name__ == "__main__":
    main()
