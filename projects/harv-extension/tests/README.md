# Test And Result Index

This directory is the human-facing index for tests and run results. Existing generated artifacts remain in `outputs/` and `run_results/` so current scripts keep working.

## Canonical Test Names

| Proposition | Canonical test name | Current artifact locations |
|---|---|---|
| Original paper pipeline | `original_paper_pipeline` | `run_results/current_intraday/`, `run_results/paper_runs/` |
| Prop 3 | `prop3_cp_pm_incremental_value` | `outputs/cp_pm_incremental_tests/`, `outputs/conditional_pm_cp_tests/` |
| Prop 4 | `prop4_cp_repo_ops_hg_18_model_suite` | `outputs/cp_repo_ops_hg_tests/`, local `outputs/cp_repo_ops_hg_actions/28269511971/` if present |
| Prop 5 | `prop5_pm_selective_overlay` | `outputs/pm_selective_overlay_tests/`, `outputs/pm_research_internal_package/prop5_overlay_exploratory/` |

## Run Bundle Rule

Every retained run should include or point to:

- README: purpose, assets, model list, config, command, and status.
- Manifest or metadata.
- Scripts or command record.
- Predictions, when generated.
- Results/statistics.
- Figures/images, when generated.
- Paper tables or LaTeX, when generated.
- Audits/logs, when relevant.

## Promotion Rule

Repo 1 can contain rough, failed, and exploratory runs. Repo 2 should contain all successful run results that the user approves for retention. Ask before promoting a run from repo 1 to repo 2. Repo 3 is not automatic; only explicit user prompts should change it.
