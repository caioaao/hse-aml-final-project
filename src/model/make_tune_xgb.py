import joblib

import pandas as pd
import optuna
from .xgb import make_xgb_loss, make_xgb_objective, train
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
    cv_splits = tscv.split(train_set['date_block_num'].values)

    objective = make_xgb_objective(make_xgb_loss(X_train, y_train, cv_splits))

    trials_db = 'sqlite:///%s' % trials_db_path
    study = optuna.create_study(direction='minimize', load_if_exists=True,
                                study_name=train_set_path, storage=trials_db)
    study.optimize(objective, n_trials=100, n_jobs=4, gc_after_trial=True)

    bst = train(study.best_params, X_train, y_train)
    bst.save_model(output_path)
