import sys
import joblib
import xgboost as xgb
from . import ClippedOutputRegressor

if __name__ == '__main__':
    output_path = sys.argv[1]
    reg = xgb.XGBRegressor(random_state=13, tree_method='gpu_hist',
                           gpu_id=0, n_jobs=-1, missing=-999)
    joblib.dump(ClippedOutputRegressor(reg), output_path)
