Note: A lot of the figures have the functionality to be tweaked for style, color, scale, etc. Pls look into the files if you need to change anything. Most of these commands should return the visuals, unless we perform the final run (with all tickers), in which case we might need to omit specifying "SPY".

Canonical run layout:
- `run_results/paper_runs/<RUN>/predictions`
- `run_results/paper_runs/<RUN>/tables`
- `run_results/paper_runs/<RUN>/figures`
- `run_results/paper_runs/<RUN>/latex`

When using the commands below, replace paths as follows:
- `run_results/current_intraday/predictions` -> `run_results/paper_runs/<RUN>/predictions`
- `run_results/current_intraday/tables` -> `run_results/paper_runs/<RUN>/tables`
- `run_results/current_intraday/figures` -> `run_results/paper_runs/<RUN>/figures`
- `run_results/current_intraday/latex` -> `run_results/paper_runs/<RUN>/latex`

Important fixed-path helpers:
- Section 2 helper writes to `run_results/current_intraday/tables/Section_2_regimes.csv` and expects predictions in `run_results/current_intraday/predictions`.
  If running manually, sync your `<RUN>/predictions` into that scratch path, run the helper, then copy the CSV back into `<RUN>/tables`.
- Section 4 figure script writes `Section_4_error_advantage_summary_*.csv` to `run_results/current_intraday/tables`.
  Copy those summaries into `<RUN>/tables` after running.
- Some sections produce both latex tables and figures. We can always choose to keep either one of those, or both. If any additional changes are required to the visuals, they can absolutely be made. All relevant visual producing files can be found under paper_export.
- We have the flexibility to choose what tickers we want to include in the resulting table/figure that is returned to us by almost every script in paper_export
- Some of these sections require helper scripts, as mentioned in the instructions below.

Overall structure:
- Main run (Intraday_benchmark.py) returns predictions and projects\harv-extension\run_results\current_intraday\tables\Section_1_table_1a_overall.csv, and runs projects\harv-extension\code\Helpers_for_paper_export_scripts\Section_5\build_section5_tables.py
- This action automatically populates a lot of feeder csvs that we need (all of them can be found under projects\harv-extension\run_results\current_intraday\tables)
- Most of the sections' results can be formed directly by running the below commands, however, some require us to manually run the helper scripts (found in projects\harv-extension\code\Helpers_for_paper_export_scripts), and then we can proceed to run the relevant paper_export script

Meaning of important types of files:
- Paper_export: Visual creating scripts
- Helpers_for_paper_export_scripts (self-explanatory)
- projects\harv-extension\run_results\current_intraday\tables: contain the feeder csvs that paper_export scripts use
- all visual results/outputs can be found in: projects\harv-extension\run_results\paper_runs\<RUN>


Post-run commands to generate results:

Directory: .../money-making\projects\harv-extension>

Section 1:

Latex table 1a: 
python code/paper_export/Section_1/latex_table/Section_1_table_1_overall.py `
  --csv run_results/current_intraday/tables/Section_1_table_1a_overall.csv `
  --out run_results/current_intraday/latex/Section_1_table_1a_overall.tex `
  --context intraday

Section 2:

Build input using:

python code/Helpers_for_paper_export_scripts/Section_2/run_regime_analysis.py

Latex table 2:
python code/paper_export/Section_2/latex_table/Section_2_table_2_regimes.py `
  --csv run_results/current_intraday/tables/Section_2_regimes.csv `
  --out run_results/current_intraday/latex/Section_2_table_2_regimes.tex `
  --include-variants

Figure 1:
(Relative)
python code/paper_export/Section_2/figure/Section_2_fig_regimes.py `
  --csv run_results/current_intraday/tables/Section_2_regimes.csv `
  --outdir run_results/current_intraday/figures

(Absolute)
python code/paper_export/Section_2/figure/Section_2_fig_regimes.py `
  --csv run_results/current_intraday/tables/Section_2_regimes.csv `
  --outdir run_results/current_intraday/figures `
  --absolute

Section 3:
Figure 2:
(Black-white):
python code/paper_export/Section_3/figure/Section_3_fig_2_temporal_stability.py `
  --pred-dir run_results/current_intraday/predictions `
  --asset SPY `
  --models HAR,PM,CP `
  --window 78 `
  --smooth ema `
  --smooth-span 39 `
  --resample W `
  --trend `
  --scale 90,170 `
  --style monochrome `
  --outdir run_results/current_intraday/figures

(Coloured)
python code/paper_export/Section_3/figure/Section_3_fig_2_temporal_stability.py `
  --pred-dir run_results/current_intraday/predictions `
  --asset SPY `
  --models HAR,PM,CP `
  --window 78 `
  --smooth ma `
  --smooth-span 13 `
  --resample W `
  --style color `
  --outdir run_results/current_intraday/figures

Section 4:
Figure 3:
(Smooth):
python code/paper_export/Section_4/render_and_figure/Section_4_render_and_fig_error_advantage.py `
  --pred-dir run_results/current_intraday/predictions `
  --asset SPY `
  --baseline HAR `
  --models PM,CP `
  --metric smape `
  --resample W `
  --smooth ema `
  --smooth-span 7 `
  --clip-pctl "1,99" `
  --scale="-50,50" `
  --outdir run_results/current_intraday/figures

(More intricate):
python code/paper_export/Section_4/render_and_figure/Section_4_render_and_fig_error_advantage.py `
  --pred-dir run_results/current_intraday/predictions `
  --asset SPY `
  --baseline HAR `
  --models PM,CP `
  --metric smape `
  --scale="-50,50" `
  --outdir run_results/current_intraday/figures

Section 5:

Helper for capacity curve:
python code/Helpers_for_paper_export_scripts/Section_5/sweep_pm_capacity.py `
  --assets SPY `
  --k-grid 3,4,5,6,7,8 `
  --n 390 `
  --warmup 200 `
  --local-dir data/market_data/clean `
  --out-csv run_results/current_intraday/tables/Section_5_fig_4_capacity_curve.csv


Figures:
 Fig 4: Capacity curve
python code/paper_export/Section_5/figure/Section_5_fig_4_capacity_curve.py `
  --csv run_results/current_intraday/tables/Section_5_fig_4_capacity_curve.csv `
  --outdir run_results/current_intraday/figures

 Fig 5: PM modifiers efficacy (uses 5a ablations CSV)
python code/paper_export/Section_5/figure/Section_5_fig_5_pm_modifiers_efficacy.py `
  --tables-dir run_results/current_intraday/tables `
  --outdir run_results/current_intraday/figures `
  --style mono

Fig 6: CP vs CP-CJ (uses 5b ablations CSV) - We might need to try a different metric as SMAPE produces no significant finding
python code/paper_export/Section_5/figure/Section_5_fig_6_cp_vs_cj_variant.py `
  --tables-dir run_results/current_intraday/tables `
  --outdir run_results/current_intraday/figures `
  --style mono `
  --metrics smape,mae,rmse,diracc


Table 3: Parameter efficiency (uses joined CSV)
python code/paper_export/Section_5/latex_table/Section_5_table_3_param_efficiency.py `
  --csv run_results/current_intraday/tables/Section_5_table_3_param_eff_joined.csv `
  --out run_results/current_intraday/latex/Section_5_table_3_param_eff.tex

Table 4: Randomized control
python code/paper_export/Section_5/latex_table/Section_5_table_4_random_controls.py `
  --in run_results/current_intraday/tables/Section_5_table_4_random_controls.csv `
  --outdir run_results/current_intraday/latex `
  --scope intraday


Section 6:

Fig 7 and table 5:

python code/paper_export/Section_6/figure_and_latex_table/Section_6_fig_7_and_table_5_asset_groups.py `
  --pred-dir run_results/current_intraday/predictions `
  --tables-dir run_results/current_intraday/tables `
  --out-tex run_results/current_intraday/latex/Section_6_table_5_asset_groups.tex `
  --out-fig run_results/current_intraday/figures/Section_6_fig_7_asset_groups.pdf `
  --baseline HAR `
  --models PM,CP `
  --auto-build

