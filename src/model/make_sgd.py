import sys

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

import optuna
import pandas as pd
import numpy as np
import joblib

from . import tscv
from ..data import df_to_X_y


MAX_EVALS = 100

DEFAULT_PARAMS = {
    'fit_intercept': False,
}
TOL = 0.0001


def _clipped_rmse(y, ypred):
    return mean_squared_error(y, np.clip(ypred, 0, 20), squared=False)


def trial_to_params(trial: optuna.Trial):
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    learning_rate = trial.suggest_categorical(
        'learning_rate', ['constant', 'optimal', 'invscaling'])

    params = {**DEFAULT_PARAMS,
              'loss': trial.suggest_categorical(
                  'loss', ['squared_loss', 'huber', 'epsilon_insensitive',
                           'squared_epsilon_insensitive']),
              'penalty': penalty,
              'alpha': trial.suggest_loguniform('alpha', 1e-7, 1.),
              'random_state': trial.suggest_int('random_state', 0, 999999),
              'learning_rate': learning_rate}

    if penalty == 'elasticnet':
        params['l1_ratio'] = trial.suggest_discrete_uniform(
            'l1_ratio', .01, .99, .01)
    if learning_rate in ['constant', 'invscaling']:
        params['eta0'] = trial.suggest_loguniform('eta0', 1e-7, 1e-1)
    if learning_rate == 'invscaling':
        params['power_t'] = trial.suggest_discrete_uniform(
            'power_t', 0.1, 0.5, 0.001)
    return params


def sgd_fit(sgd, X_train, y_train, X_val, y_val, early_stop_rounds=10,
            tol=0.0001, callback=None, max_iter=200):
    best_loss = np.inf
    best_iter_n = 0
    for iter_n in range(max_iter):
        sgd.partial_fit(X_train, y_train)
        intermediate_value = _clipped_rmse(y_val, sgd.predict(X_val))
        print(' iter %2d | val-rmse %8.5f' % (iter_n, intermediate_value))
        if callback:
            callback(sgd, iter_n, intermediate_value)
        if best_loss > intermediate_value + tol:
            best_iter_n = iter_n
            best_loss = intermediate_value

        if iter_n - best_iter_n > early_stop_rounds:
            break
    return sgd


def _make_objective(X_train, y_train, X_val, y_val, early_stop_rounds=10):
    def objective(trial: optuna.Trial):
        def fit_callback(sgd, iter_n, intermediate_value):
            trial.report(intermediate_value, iter_n)
            if trial.should_prune():
                raise optuna.TrialPruned()

        sgd = SGDRegressor(**trial_to_params(trial))
        sgd = sgd_fit(sgd, X_train, y_train, X_val, y_val,
                      callback=fit_callback)

        return _clipped_rmse(y_val, sgd.predict(X_val))

    return objective


if __name__ == '__main__':
    trials_db_path = sys.argv[1]
    train_set = pd.read_parquet(sys.argv[2])
    preprocessor = joblib.load(sys.argv[3])
    output_path = sys.argv[4]

    X, y = df_to_X_y(train_set)

    X_train, y_train, X_val, y_val = tscv.train_test_split(
        X, y, date_vec=train_set['date_block_num'].values,
        train_start=16)
    del train_set

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)

    trials_db = 'sqlite:///%s' % trials_db_path
    study = optuna.create_study(
        direction='minimize', load_if_exists=True, study_name=output_path,
        storage=trials_db, pruner=optuna.pruners.HyperbandPruner())

    n_trials = MAX_EVALS - len(study.trials)
    if n_trials > 0:
        study.optimize(_make_objective(X_train, y_train, X_val, y_val),
                       n_trials=n_trials, n_jobs=1,
                       gc_after_trial=True)

    sgd = SGDRegressor(**DEFAULT_PARAMS, **study.best_params)
    print('Finding optimal max_iter')
    sgd = sgd_fit(sgd, X_train, y_train, X_val, y_val, max_iter=1000,
                  early_stop_rounds=50)
    print('Optimal max_iter: %3d' % sgd.n_iter_)

    sgd = SGDRegressor(**DEFAULT_PARAMS, **study.best_params,
                       early_stopping=False, max_iter=sgd.n_iter_)

    print('Final estimator: %s' % sgd)
    reg = Pipeline([('pre', preprocessor),
                    ('sgd', sgd)])
    print('Fitting final estimator')
    reg.fit(X, y)
    joblib.dump(reg, output_path)
