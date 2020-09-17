import sys

import joblib
import xgboost as xgb
import pandas as pd

from . import ClippedOutputRegressor
from ..data import df_to_X_y


def make_reg():
    base_reg = xgb.XGBRegressor(random_state=13, tree_method='gpu_hist',
                                gpu_id=0, n_jobs=-1, missing=-999)
    return ClippedOutputRegressor(base_reg)


if __name__ == '__main__':
    output_path = sys.argv[-1]

    reg = make_reg()

    if len(sys.argv) > 1:
        train_set = pd.read_parquet(sys.argv[1])
        X, y = df_to_X_y(train_set)
        print("Fitting baseline model")
        reg = reg.fit(X, y)

    joblib.dump(reg, output_path)
