import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.reporting_utils import aggregate_regimes_from_predictions, save_table_regimes

pred_dir = 'code/outputs/intraday/predictions'
models   = ['HAR','HAR_J','HAR_CJ','HAR_TCJ','PM','PM_VW','PM_AD','CP','CP_CJ','EXH','HAM','RAND']
df = aggregate_regimes_from_predictions(pred_dir, models, qlow=0.33, qhigh=0.66)
os.makedirs('code/outputs/intraday/tables', exist_ok=True)
save_table_regimes(df, 'code/outputs/intraday/tables/Section_2_regimes.csv')
print('Wrote code/outputs/intraday/tables/Section_2_regimes.csv')
