import sys
import joblib
import xgboost as xgb

if __name__ == '__main__':
    output_path = sys.argv[1]
    reg = xgb.XGBRegressor(random_state=13, tree_method='gpu_hist',
                           gpu_id=0, n_jobs=-1)
    joblib.dump(reg, output_path)
