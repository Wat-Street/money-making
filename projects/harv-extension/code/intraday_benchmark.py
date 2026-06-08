import sys, os, argparse, time, gc, traceback
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from utils.data_utils import fetch_intraday_data, calculate_intraday_realized_volatility, fit_and_predict_extended
from utils.reporting_utils import aggregate_overall_from_predictions, save_table_overall
from utils.harvey_utils import add_harv_terms, add_harv_j_terms, add_harv_cj_terms, add_harv_tcj_terms
from utils.models_utils import (
    add_exhaustive_terms, add_hamming_terms,
    add_prime_modulo_terms, add_volume_weighted_prime_modulo_terms,
    add_volume_weighted_adaptive_prime_modulo_terms,
    contig_prime_modulo, contig_prime_modulo_with_jumps,
    random_sets, contiguous_random_sets
)

MODEL_FUNCS = {
    'HAR': add_harv_terms,
    'HAR_J': add_harv_j_terms,
    'HAR_CJ': add_harv_cj_terms,
    'HAR_TCJ': add_harv_tcj_terms,
    'PM': add_prime_modulo_terms,
    'PM_VW': add_volume_weighted_prime_modulo_terms,
    'PM_AD': add_volume_weighted_adaptive_prime_modulo_terms,
    'CP': contig_prime_modulo,
    'CP_CJ': contig_prime_modulo_with_jumps,
    'EXH': add_exhaustive_terms,
    'HAM': add_hamming_terms,
    'RAND': random_sets,
    'CRS': contiguous_random_sets,
}

class GracefulStop(Exception):
    """Signal a graceful, intentional stop (e.g., OOM) so main can exit cleanly."""
    pass

def _is_oom_like(err: Exception) -> bool:
    """Detects common OOM/BLAS allocation failures we saw (MemoryError, 'Unable to allocate', 'gesdd failed')."""
    s = (str(err) or "").lower()
    return (
        isinstance(err, MemoryError)
        or "unable to allocate" in s
        or "init_gesdd failed" in s
        or "svd did not converge" in s and "gesdd" in s
        or "out of memory" in s
    )

def _signal_stop(outdir: str, ticker: str, model: str, err: Exception):
    """Write a stop marker so you know exactly why/where it halted."""
    os.makedirs(outdir, exist_ok=True)
    marker = os.path.join(outdir, "FATAL_STOP.txt")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(marker, "a", encoding="utf-8") as f:
        f.write(f"[{now}] HALT ticker={ticker} model={model} error={type(err).__name__}: {err}\n")
        tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
        f.write(tb + "\n")
    print(f"[FATAL] OOM-like error at {ticker}/{model}. Wrote marker: {marker}", flush=True)

def _default_models(include_variants: bool = True):
    cores = ['HAR','HAR_J','HAR_CJ','HAR_TCJ','PM','CP','EXH','HAM','RAND', 'CRS']
    variants = ['PM_VW','PM_AD','CP_CJ']
    return cores + (variants if include_variants else [])

def discover_local_tickers(base_dir: str) -> list:
    if not os.path.isdir(base_dir):
        return []
    return sorted([f.replace('_5m.csv','') for f in os.listdir(base_dir) if f.endswith('_5m.csv')])

def _fmt_s(sec: float) -> str:
    if sec >= 3600:
        h = int(sec // 3600); m = int((sec % 3600) // 60)
        return f"{h}h{m}m"
    if sec >= 60:
        m = int(sec // 60); s = int(sec % 60)
        return f"{m}m{s:02d}s"
    return f"{sec:.1f}s"

def _append_line(path: str, line: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")

def _append_model_checkpoint(out_csv: str, f: pd.DataFrame, model_name: str):
    """
    Append/merge this model's columns into the ticker predictions CSV progressively.
    - Keeps 'Actual' only once (prefers existing file's Actual if present)
    - Drops any prior columns for this model so new ones overwrite cleanly
    - Uses OUTER join so variants with shorter aligned windows do not truncate
      previously checkpointed baseline predictions
    """
    new = f.copy()
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if os.path.exists(out_csv):
        acc = pd.read_csv(out_csv, parse_dates=['Date'])
        if 'Date' not in acc.columns:
            acc = acc.rename(columns={'index': 'Date'})
        acc = acc.set_index('Date')

        new_actual = new['Actual'].copy() if 'Actual' in new.columns else None
        if 'Actual' in acc.columns and 'Actual' in new.columns:
            new = new.drop(columns=['Actual'])

        model_cols_new = list(new.columns)
        model_cols_existing = [c for c in model_cols_new if c in acc.columns]
        if model_cols_existing:
            acc = acc.drop(columns=model_cols_existing)

        joined = acc.join(new, how='outer')
        if new_actual is not None and 'Actual' in joined.columns:
            joined['Actual'] = joined['Actual'].combine_first(new_actual.reindex(joined.index))
    else:
        joined = new

    out_df = joined.reset_index().rename(columns={'index': 'Date'})
    out_df.to_csv(out_csv, index=False)
    print(f"[CHECKPOINT] appended {model_name} -> {out_csv} (rows={len(out_df)}, cols={len(out_df.columns)})", flush=True)

def _completed_models_from_checkpoint(out_csv: str, models: list) -> set:
    if not os.path.exists(out_csv):
        return set()
    try:
        header = pd.read_csv(out_csv, nrows=0).columns
    except Exception:
        return set()
    return {m for m in models if f"Predicted_{m}" in header}

def run_for_ticker(ticker: str, models: list, n: int, warmup: int, local_dir: str, outdir: str,
                   timing_log_path: str, resume: bool = False) -> float:
    t0 = time.perf_counter()
    pred_dir = os.path.join(outdir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    out_csv = os.path.join(pred_dir, f'{ticker}.csv')
    completed_models = _completed_models_from_checkpoint(out_csv, models) if resume else set()
    if completed_models:
        done = ",".join([m for m in models if m in completed_models])
        print(f"[RESUME] [{ticker}] checkpoint has completed models: {done}", flush=True)

    print(f"[STEP] [{ticker}] fetch_intraday_data...", flush=True)
    s = time.perf_counter()
    raw = fetch_intraday_data(ticker, use_local=True, local_dir=local_dir)
    print(f"[STEP] [{ticker}] fetch_intraday_data done in {_fmt_s(time.perf_counter()-s)}", flush=True)

    print(f"[STEP] [{ticker}] calculate_intraday_realized_volatility...", flush=True)
    s = time.perf_counter()
    vol = calculate_intraday_realized_volatility(raw)
    print(f"[STEP] [{ticker}] RV computed in {_fmt_s(time.perf_counter()-s)}", flush=True)

    frames = []
    per_model_times = []

    for m in models:
        if m in completed_models:
            print(f"[RESUME] [{ticker}] {m} already checkpointed; skipping", flush=True)
            per_model_times.append((m, 0.0))
            continue

        print(f"[MODEL] [{ticker}] {m} start", flush=True)
        ms = time.perf_counter()
        try:
            func = MODEL_FUNCS[m]
            extended = func(vol.copy(), n)
            # Include all engineered feature families (RV_*, PM_*, CP_*)
            features = [c for c in extended.columns if c.startswith(('RV', 'PM', 'CP'))]
            preds = fit_and_predict_extended(extended, features, n, warmup, model_name=m)
            if preds is None or preds.empty:
                print(f"[MODEL] [{ticker}] {m} produced no predictions (skipping)", flush=True)
                per_model_times.append((m, 0.0))
                # free intermediates
                try:
                    del extended, preds, features
                except NameError:
                    pass
                gc.collect()
                continue

            if 'Predicted' in preds.columns and f'Predicted_{m}' not in preds.columns:
                preds[f'Predicted_{m}'] = preds['Predicted']
            cols = ['Actual', f'Predicted_{m}', f'Err_{m}', f'AbsErr_{m}', f'SMAPE_{m}_pct']
            f = preds[[c for c in cols if c in preds.columns]]
            frames.append(f)

            # checkpoint after each model
            _append_model_checkpoint(out_csv, f, m)

            m_dur = time.perf_counter() - ms
            per_model_times.append((m, m_dur))
            print(f"[MODEL] [{ticker}] {m} done in {_fmt_s(m_dur)}", flush=True)

        except Exception as e:
            if _is_oom_like(e):
                _signal_stop(outdir, ticker, m, e)
                # keep what we have; do NOT write any merged overwrite
                now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                _append_line(timing_log_path, f"[{now}] TICKER {ticker} HALT at model {m}: {type(e).__name__}: {e}")
                raise GracefulStop()  # bubble to main
            else:
                # Unknown fatal error: propagate (so it fails loudly)
                raise
        finally:
            # free intermediates aggressively each model to reduce RAM
            try:
                del extended, preds, features
            except NameError:
                pass
            gc.collect()

    if not frames:
        if resume and os.path.exists(out_csv):
            print(f"[RESUME] [{ticker}] no new models needed; preserving {out_csv}", flush=True)
        else:
            print(f"[WARN] [{ticker}] No model produced frames; skipping save.", flush=True)
            return time.perf_counter() - t0

    if frames and not resume:
        # --- safe merge: keep 'Actual' only once ---
        merged = frames[0].copy()
        for f in frames[1:]:
            f2 = f.drop(columns=[c for c in ['Actual'] if c in f.columns])
            merged = merged.join(f2, how='inner')

        merged = merged.reset_index().rename(columns={'index':'Date'})

        print(f"[SAVE ] [{ticker}] writing predictions CSV...", flush=True)
        s = time.perf_counter()
        merged.to_csv(out_csv, index=False)
        print(f"[SAVE ] [{ticker}] wrote {out_csv} in {_fmt_s(time.perf_counter()-s)} (rows={len(merged)})", flush=True)
    elif frames:
        print(f"[SAVE ] [{ticker}] resume mode: preserving checkpoint-merged {out_csv}", flush=True)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    _append_line(timing_log_path, f"[{now}] TICKER {ticker} model timings:")
    for m, secs in per_model_times:
        _append_line(timing_log_path, f"  - {m}: {secs:.3f}s")

    elapsed = time.perf_counter() - t0
    print(f"[TICKER] [{ticker}] total {_fmt_s(elapsed)}", flush=True)
    _append_line(timing_log_path, f"[{now}] TICKER {ticker} total: {elapsed:.3f}s")
    return elapsed

def main():
    p = argparse.ArgumentParser(description="Intraday benchmark (local 5m CSVs)")
    p.add_argument('--tickers', type=str, default='', help='Comma-separated. If empty, auto-discover in local_dir.')
    p.add_argument('--local-dir', type=str, default='Datasets/clean')
    p.add_argument('--outdir', type=str, default='code/outputs/intraday')
    p.add_argument('--models', type=str, default='', help='Comma-separated model codes; if empty we choose defaults')
    p.add_argument('--n', type=int, default=22)
    p.add_argument('--warmup', type=int, default=600)
    p.add_argument('--require-explicit', action='store_true', help='Error if --tickers not provided (no auto-discovery).')
    p.add_argument('--skip-post', action='store_true', help='Skip overall/table aggregation and Section 5 post-run build.')
    p.add_argument('--skip-section5', action='store_true', help='Skip Section 5 post-run table build (overall table still produced).')
    p.add_argument('--resume', action='store_true', help='Skip models whose Predicted_<MODEL> columns already exist in prediction checkpoints.')
    p.add_argument('--include-variants', dest='include_variants', action='store_true', default=True,
                  help='Include PM_VW, PM_AD, CP_CJ when models not explicitly specified (default: on)')
    p.add_argument('--no-variants', dest='include_variants', action='store_false')
    args = p.parse_args()

    if args.models.strip():
        models = [m.strip() for m in args.models.split(',') if m.strip()]
    else:
        models = _default_models(include_variants=args.include_variants)

    tickers = [t.strip() for t in args.tickers.split(',') if t.strip()]
    if not tickers:
        if args.require_explicit:
            raise SystemExit("No --tickers provided and --require-explicit set. Pass a comma-separated list via --tickers.")
        tickers = discover_local_tickers(args.local_dir)

    os.makedirs(os.path.join(args.outdir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'tables'), exist_ok=True)

    timing_log_path = os.path.join(args.outdir, 'timing_progress.log')
    per_ticker_csv = os.path.join(args.outdir, 'tables', 'timing_by_ticker.csv')
    if not os.path.exists(per_ticker_csv):
        pd.DataFrame(columns=["timestamp_utc","ticker","seconds","hhmm"]).to_csv(per_ticker_csv, index=False)

    overall_start = time.perf_counter()
    elapsed_list = []

    try:
        for idx, t in enumerate(tickers, start=1):
            print(f"[INTRA] {t}", flush=True)
            elapsed = run_for_ticker(t, models, args.n, args.warmup, args.local_dir, args.outdir, timing_log_path, resume=args.resume)
            elapsed_list.append(elapsed)
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            hhmm = _fmt_s(elapsed)
            pd.DataFrame([{"timestamp_utc": now, "ticker": t, "seconds": round(elapsed,3), "hhmm": hhmm}]).to_csv(
                per_ticker_csv, mode="a", header=False, index=False
            )
            avg_per_ticker = sum(elapsed_list) / len(elapsed_list)
            est_total_45 = avg_per_ticker * 45
            so_far = time.perf_counter() - overall_start
            est_remaining_45 = max(est_total_45 - so_far, 0)
            print(
                f"[ETA  ] avg/ticker ~ {_fmt_s(avg_per_ticker)} -> 45 tickers ~ {_fmt_s(est_total_45)} ",
                f"(remaining if aiming for 45: {_fmt_s(est_remaining_45)})",
                flush=True
            )

    except GracefulStop:
        # Stop immediately; keep whatever is already on disk from checkpoints.
        print("[HALT] Graceful stop requested due to OOM-like error. Preserving all checkpoints.", flush=True)
        return

    if not args.skip_post:
        # Post-run aggregation (only if we completed without a graceful stop)
        overall, _ = aggregate_overall_from_predictions(os.path.join(args.outdir, 'predictions'), models)
        out_csv = os.path.join(args.outdir, 'tables', 'Section_1_table_1a_overall.csv')
        save_table_overall(overall, out_csv)
        print(f"[INTRA] wrote {out_csv}", flush=True)

        if not args.skip_section5:
            try:
                import subprocess  # keep os/sys imports at top-level
                pred_dir = os.path.join(args.outdir, 'predictions')
                tables_dir = os.path.join(args.outdir, 'tables')
                models_csv = ",".join(models)
                cmd = [
                    sys.executable, 'code/Helpers_for_paper_export_scripts/Section_5/build_section5_tables.py',
                    '--pred-dir', pred_dir,
                    '--tables-dir', tables_dir,
                    '--models', models_csv,
                    '--rand-baseline', 'PM'
                ]
                print("[post] building Section 5 tables…")
                subprocess.run(cmd, check=True)
            except Exception as e:
                print(f"[post] skipped Section 5 build: {e}")

if __name__ == "__main__":
    main()
