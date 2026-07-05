# Agent Operating Rules For HARV Repos

This file is for Codex/agent work on the HAR/RV extension project. Follow these rules before classifying, moving, or promoting model code or run results.

## Repo Roles

- Repo 1: `Harman6139/money-making` / local rough work. Use this for experiments, failed runs, scratch runs, and implementation testing.
- Repo 2: `Wat-Street/money-making` branch `cp-pm-incremental-actions`. Use this for model implementation, diagnostics, and successful run results that the user explicitly approves for retention.
- Repo 3: `Wat-Street/money-making` branch `pre-final`. Use this only for draft-ready paper inputs, finalized data, and manually requested paper changes. Do not promote anything here unless the user explicitly asks.

## Required Clarification Before Promotion

Before moving or copying anything between repo roles, ask the user:

- Is this a model implementation change, a test result, a paper asset, or scratch output?
- If it is a test result, is it failed, rough-but-worth-keeping, successful-but-unapproved, or approved for repo 2?
- Should any part of this touch repo 3? Repo 3 changes require an explicit prompt.

Approved tests from repo 1 go to repo 2 only. Repo 3 is manual and explicit.

## Model Sync Rule

Repo 1 and repo 2 should always carry the same model implementations and model documentation. Repo 3 may contain only the draft-ready subset.

Use `models/` as the human-facing model inventory. Add a child folder for each new named model or stable model family. Do not create generic `v1`, `v2`, etc. folders. If a model changes, describe the change in the model name or README, for example `CP_REPO_FRESH` versus older `CP`.

Important CP naming:

- `CP` is the older original contiguous-prime model used by the original paper pipeline.
- `CP_REPO_FRESH` is the corrected fresh repo-CP benchmark used as the Prop 4 denominator.

## Test Naming Rule

Use proposition-aware names:

- Prop 3: CP+PM incremental value tests.
- Prop 4: CP-REPO-OPS-HG 18-model suite.
- Prop 5: PM/path-order selective overlay tests.

Each kept run should have:

- `README.md` with purpose, models, assets, command/config, and interpretation status.
- `manifest.json` or equivalent metadata when available.
- `scripts/` or a command reference sufficient to rerun.
- `predictions/` when prediction panels were produced.
- `results/` for calculated statistics such as SMAPE, MAE, RMSE, win rate, audits, and diagnostics.
- `figures/` or `images/` when produced.
- `paper_tables/` or `latex/` when paper-ready summaries were produced.

Keep generated output paths stable unless you update every script and README that references them.
