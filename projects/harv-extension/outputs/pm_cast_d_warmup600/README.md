# PM-CAST-D Warmup-600 Validation

This folder contains the repo-1 implementation and run outputs for the
practitioner application test of PM-native averaging.

The model is selected exclusively on pre-OOS origins, then evaluated against
fresh `CP_REPO_FRESH` predictions on all ten assets. Warmup `600` is both the
CLI default and a hard run invariant.

## Local Smoke

```powershell
python outputs/pm_cast_d_warmup600/scripts/run_pm_cast_d.py --mode tune --assets AAPL --warmup 600 --outdir outputs/pm_cast_d_warmup600/local_smoke/tuning
python outputs/pm_cast_d_warmup600/scripts/run_pm_cast_d.py --mode run-asset --asset AAPL --warmup 600 --config outputs/pm_cast_d_warmup600/local_smoke/tuning/frozen_config.json --outdir outputs/pm_cast_d_warmup600/local_smoke/AAPL
```

The full run is dispatched through `.github/workflows/pm_cast_d_warmup600.yml`.
No result from this folder is promoted to repo 2 or repo 3 without explicit
approval.

Latest verified run: `verified_actions_29069772802/README.md`. It improves
SMAPE and MAE with positive moving-block intervals but does not beat CP on
MSE/RMSE, so it is retained as a completed needs-review result rather than an
all-fronts success.
