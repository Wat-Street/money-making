# Agent Notes

This repository is repo 1 when used from `Harman6139/money-making` and repo 2 when the same branch is pushed to `Wat-Street/money-making`.

For HAR-RV / HARV extension work, read `projects/harv-extension/AGENTS.md` before moving models, results, or paper assets.

Core rules:

- Repo 1 (`Harman6139/money-making`, branch `cp-pm-incremental-actions`) is the rough construction, GitHub Actions, diagnostics, and raw-artifact space.
- Repo 2 (`Wat-Street/money-making`, branch `cp-pm-incremental-actions`) is the consolidated working layer. Keep only scripts, cleaned results, review notes, and approved run outputs here.
- Repo 3 (`Wat-Street/pre-final`) is manual and draft-facing; never promote files there unless explicitly asked.
- Ask the user before classifying or promoting a run result across repos.
- Keep model folders descriptive. Do not create generic `v1`, `v2`, etc. names.
- Prop 3 is CP+PM incremental value. Prop 4 is the CP-REPO-OPS-HG 18-model suite. Prop 5 is the PM selective overlay line.

## CP/PM Benchmark Contract for New Model Runs

- Use `n=22`.
- Correct benchmark is always `CP_REPO_FRESH`.
- `CP_REPO_FRESH = RV/HAR + contig_prime_modulo(vol.copy(), n=22, per_day_normalize=False)`.
- `CP_REPO_FRESH` feature set is repo `RV*` columns plus repo `CP_*` columns only.
- `CP_REPO_FRESH` must exclude `CPB_B*`, `PM_*`, OPS, shape, gates, and placebo features.
- Correct PM construction is `add_prime_modulo_terms(..., n=22)`, using the repo minimal-prime construction; for `n=22` expect primes including `2, 3, 5`.
- `CPB_B*` four-block lag means are diagnostic only. They are not the benchmark and must not define paper/result comparisons.
- All models must compare against fresh `CP_REPO_FRESH` on identical assets, timestamps, targets, rows, warmup, and forecast origins.
- No global standardization or lookahead. Scaling, gates, ridge choices, validation splits, and feature construction must use only information available before the forecast target.
- Use deterministic seeds, normally `--seed 42`, for random residues, shuffled lags, random gates, and any new stochastic model.

## Run Settings

- Historical corrected Prop 4 run `28269511971` used the runner default `--warmup 600`; keep those outputs separate when reproducing old tables.
- For Harman's next additional/new-model tests, use explicit `--warmup 200` and a separate output root. Do not mix warmup-200 results into the existing Prop 4 review package.
- Full asset universe is `AAPL,AMZN,EEM,FXI,GLD,GOOGL,HYG,QQQ,SPY,TLT`.
- Smoke tests should use one asset, usually `AAPL`.
- Run from `projects/harv-extension`.

Smoke example for new-model/warmup-200 tests:

```powershell
python outputs/cp_repo_ops_hg_tests/scripts/run_cp_repo_ops_hg_tests.py --mode smoke --assets AAPL --phase all --n 22 --warmup 200 --seed 42 --force --output-root outputs/cp_repo_ops_hg_new_model_tests_warmup200 --audit-no-lookahead --audit-repo-cp-reproduction --audit-fast-slow-equivalence --cv-mode inner --lambda-grid 0.0001,0.001,0.01 --lambda-r-ratio-grid 2,5,10,20
```

Full example for new-model/warmup-200 tests:

```powershell
python outputs/cp_repo_ops_hg_tests/scripts/run_cp_repo_ops_hg_tests.py --mode full --phase all --n 22 --warmup 200 --seed 42 --force --output-root outputs/cp_repo_ops_hg_new_model_tests_warmup200 --audit-no-lookahead --audit-repo-cp-reproduction --audit-fast-slow-equivalence --cv-mode inner --lambda-grid 0.0001,0.001,0.01 --lambda-r-ratio-grid 2,5,10,20
```

Regenerate tables from existing outputs:

```powershell
python outputs/cp_repo_ops_hg_tests/scripts/run_cp_repo_ops_hg_tests.py --mode tables-only --outdir outputs/cp_repo_ops_hg_new_model_tests_warmup200
```

## Expected Output Contract

- Predictions by asset under `predictions/smoke/` or `predictions/full/`.
- `results/model_overall_by_asset.csv`
- `results/model_conditional_by_asset.csv`
- `results/model_conditional_pooled.csv`
- `results/ablation_ladder_summary.csv`
- `results/placebo_summary.csv`
- `results/model_feature_manifest.csv`
- `results/run_metadata.csv`
- `results/run_failures.csv`
- `paper_tables/main_model_summary.csv`
- `paper_tables/conditional_summary.csv`
- `paper_tables/placebo_summary.csv`
- `figures/ablation_ladder.png`
- `figures/conditional_advantages.png`
- `README.md`
- Smoke/implementation markers must not be called `SUCCESS.txt`; reserve `SUCCESS.txt` for a completed full paper-ready run.

Every result row must carry the benchmark name, asset, condition, observation count, model and CP SMAPE, mean/median/win-rate advantage vs CP, model and CP absolute error, and absolute-error advantage vs CP.

When adding new models, append them to the runner registry/order without replacing `CP_REPO_FRESH`. New result folders must document model features, eligibility, no-lookahead checks, and whether the model is a paper candidate, exploratory challenger, or placebo.
