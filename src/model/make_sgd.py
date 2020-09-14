import sys

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline

from optuna.integration import OptunaSearchCV
import optuna
import pandas as pd
import numpy as np
import joblib

from . import tscv
from ..data import df_to_X_y


MAX_EVALS = 100

PARAMS_DISTRIBUTIONS = {
    'alpha': optuna.distributions.LogUniformDistribution(1e-5, 1e-1)
}


def _clipped_rmse(y, ypred):
    return mean_squared_error(y, np.clip(ypred, 0, 20), squared=False)


_clipped_rmse_score = make_scorer(_clipped_rmse, greater_is_better=False)

if __name__ == '__main__':
    trials_db_path = sys.argv[1]
    train_set = pd.read_parquet(sys.argv[2])
    preprocessor = joblib.load(sys.argv[3])
    output_path = sys.argv[4]

    X_train, y_train = df_to_X_y(train_set)
    cv_splits = tscv.split(train_set['date_block_num'].values, n=1, window=16)
    del train_set

    trials_db = 'sqlite:///%s' % trials_db_path
    study = optuna.create_study(
        direction='maximize', load_if_exists=True, study_name=output_path,
        storage=trials_db, pruner=optuna.pruners.HyperbandPruner())

    sgd_cv = OptunaSearchCV(
        SGDRegressor(fit_intercept=False, random_state=38),
        param_distributions=PARAMS_DISTRIBUTIONS,
        cv=cv_splits, max_iter=1000, scoring=_clipped_rmse_score,
        n_trials=MAX_EVALS, enable_pruning=True, study=study, verbose=2)

    reg_cv = Pipeline([('pre', preprocessor),
                       ('sgd_cv', sgd_cv)])

    n_trials = MAX_EVALS - len(study.trials)
    if n_trials > 0:
        reg_cv.fit(X_train, y_train)
    print('CV: %s' % reg_cv)

    reg = Pipeline([('pre', preprocessor),
                    ('sgd', reg_cv.named_steps['sgd_cv'].best_estimator_)])
    print(reg)

    joblib.dump(reg, output_path)
