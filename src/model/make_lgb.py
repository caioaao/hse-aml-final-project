import sys

import pandas as pd
import joblib
import optuna
import optuna.integration.lightgbm as optuna_lgb
import lightgbm as lgb

from .tscv import train_test_split
from ..feature_engineering import df_to_X_y

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "nthreads": -1
}

if __name__ == '__main__':
    trials_db_path = sys.argv[1]
    train_set = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    trials_db = 'sqlite:///%s' % trials_db_path

    X, y = df_to_X_y(train_set)
    X_train, y_train, X_val, y_val = train_test_split(
        X, y, date_vec=train_set['date_block_num'].values)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)

    params = {**DEFAULT_PARAMS}

    study = optuna.create_study(
        load_if_exists=True, study_name=output_path,
        storage=trials_db)

    model = optuna_lgb.train(params, dtrain, valid_sets=[dtrain, dval],
                             early_stopping_rounds=100, verbose_eval=10,
                             study=study)

    print('Params: %s' % str(model.params))
    print('Best iteration: %d' % model.best_iteration)

    joblib.dump(model, output_path)
