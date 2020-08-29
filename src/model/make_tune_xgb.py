import joblib

import pandas as pd
import optuna

from .xgb import (make_xgb_loss, make_xgb_objective, best_num_round,
                  sklearn_regressor)
from . import tscv
from ..feature_engineering import df_to_X_y


MAX_EVALS = 200


if __name__ == '__main__':
    import sys
    trials_db_path = sys.argv[1]
    train_set_path = sys.argv[2]
    output_path = sys.argv[3]

    train_set = pd.read_parquet(train_set_path)
    X_train, y_train = df_to_X_y(train_set)
    cv_splits = tscv.split(train_set['date_block_num'].values, n=1)

    objective = make_xgb_objective(make_xgb_loss(X_train, y_train, cv_splits))

    trials_db = 'sqlite:///%s' % trials_db_path
    study = optuna.create_study(
        direction='minimize', load_if_exists=True, study_name=train_set_path,
        storage=trials_db, sampler=optuna.samplers.RandomSampler(seed=8338),
        pruner=optuna.pruners.HyperbandPruner())

    try:
        study.optimize(objective, n_trials=50, n_jobs=4, gc_after_trial=True)
    except KeyboardInterrupt:
        print("Canceling optimization step before it finishes")

    best_ntree_limit = best_num_round(study.best_params, X_train, y_train,
                                      cv_splits)

    reg = sklearn_regressor(study.best_params, best_ntree_limit)

    reg = reg.fit(X_train, y_train)

    print(reg)
    joblib.dump(reg, output_path)
