import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.reporting_utils import aggregate_regimes_from_predictions, save_table_regimes

pred_dir = 'run_results/current_intraday/predictions'
models   = ['HAR','HAR_J','HAR_CJ','HAR_TCJ','PM','PM_VW','PM_AD','CP','CP_CJ','EXH','HAM','RAND','CRS']
df = aggregate_regimes_from_predictions(pred_dir, models, qlow=0.33, qhigh=0.66)
os.makedirs('run_results/current_intraday/tables', exist_ok=True)
save_table_regimes(df, 'run_results/current_intraday/tables/Section_2_regimes.csv')
print('Wrote run_results/current_intraday/tables/Section_2_regimes.csv')
