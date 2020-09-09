import joblib

import pandas as pd
import optuna
from optuna.integration import XGBoostPruningCallback

import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from optuna import Trial
from sklearn.metrics import mean_squared_error


from . import tscv
from ..feature_engineering import df_to_X_y


MAX_EVALS = 50


DEFAULT_PARAMS = {"n_jobs": -1,
                  "objective": "reg:squarederror"}


def _xgb_feval(y_pred, dtrain):
    try:
        result = mean_squared_error(
            dtrain.get_label(), np.clip(y_pred, 0, 20),
            squared=False)
    except ValueError:
        result = np.nan
    return 'clipped-rmse', result


def _trial_to_params(trial: Trial):
    params = {**DEFAULT_PARAMS,
              "booster": trial.suggest_categorical(
                  "booster", ['gbtree']), # , 'gblinear', 'dart' are too slow
              "seed": trial.suggest_int('seed', 0, 999999),
              "learning_rate": trial.suggest_loguniform(
                  'learning_rate', 0.005, 0.5),
              "reg_alpha": trial.suggest_categorical(
                  "reg_alpha", [0, 0, 0, 0, 0, 0.00000001,
                                0.00000005, 0.0000005, 0.000005]),
              "reg_lambda": trial.suggest_categorical(
                  "reg_lambda", [1, 1, 1, 1, 2, 3, 4, 5, 1])}

    if params['booster'] == 'gbtree' or params['booster'] == 'dart':
        params.update({
            "max_depth": trial.suggest_int('max_depth', 2, 25),
            "subsample": trial.suggest_discrete_uniform('subsample',
                                                        .2, 1, .05),
            "colsample_bytree": trial.suggest_discrete_uniform(
                'colsample_bytree', .20, 1., .01),
            "colsample_bylevel": trial.suggest_discrete_uniform(
                'colsample_bylevel', .20, 1., .01),
            "colsample_bynode": trial.suggest_discrete_uniform(
                'colsample_bynode', .20, 1., .01),
            "gamma": trial.suggest_categorical("gamma", [0, 0, 0, 0, 0, 0.01,
                                                         0.1, 0.2, 0.3, 0.5,
                                                         1., 10., 100.]),
            "min_child_weight": trial.suggest_categorical('min_child_weight',
                                                          [1, 1, 1, 1, 2, 3,
                                                           4, 5, 1, 6, 7, 8, 9,
                                                           10, 11, 15, 30, 60,
                                                           100, 1, 1, 1]),
            "max_delta_step": trial.suggest_categorical("max_delta_step",
                                                        [0, 0, 0, 0, 0,
                                                         1, 2, 5, 8]),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]),
            "tree_method": "gpu_hist",
            "gpu_id": 0})
    if params["booster"] == "dart":
        params.update({
            "sample_type": trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"]),
            "normalize_type": trial.suggest_categorical(
                "normalize_type", ["tree", "forest"]),
            "rate_drop": trial.suggest_loguniform("rate_drop", 1e-8, 1.0),
            "skip_drop": trial.suggest_loguniform("skip_drop", 1e-8, 1.0)})
    return params


def make_xgb_loss(X_train, y_train, cv_splits, verbose=True):
    dtrain = xgb.DMatrix(X_train, y_train)

    def loss(params, callbacks=[]):
        return xgb.cv(
            params, dtrain,
            callbacks=callbacks,
            folds=cv_splits, verbose_eval=verbose,
            feval=_xgb_feval, maximize=False, num_boost_round=1000,
            early_stopping_rounds=10
        )['test-clipped-rmse-mean'].min()

    return loss


def make_xgb_objective(xgb_loss):
    return lambda trial: xgb_loss(
        _trial_to_params(trial), callbacks=[
            XGBoostPruningCallback(
                trial, observation_key='test-clipped-rmse')])


def _complete_params(params):
    params = {**DEFAULT_PARAMS, **params}
    if params['booster'] == 'gbtree':
        params.update({"tree_method": "gpu_hist",
                       "gpu_id": 0})
    return params


def train(params, X_train, y_train):
    dtrain = xgb.DMatrix(X_train, y_train)
    return xgb.train(_complete_params(params), dtrain)


def best_num_round(params, X, y, cv_splits, verbose=True):
    params = _complete_params(params)
    train_idx, test_idx = cv_splits[-1]
    dtrain = xgb.DMatrix(X[train_idx], y[train_idx])
    dtest = xgb.DMatrix(X[test_idx], y[test_idx])
    bst = xgb.train(params, dtrain, early_stopping_rounds=50,
                    num_boost_round=1000, evals=[(dtrain, 'dtrain'),
                                                 (dtest, 'dtest')],
                    feval=_xgb_feval, verbose_eval=verbose)
    return bst.best_ntree_limit


def sklearn_regressor(params, num_round):
    return XGBRegressor(n_estimators=num_round, **_complete_params(params))


if __name__ == '__main__':
    import sys
    trials_db_path = sys.argv[1]
    train_set_path = sys.argv[2]
    output_path = sys.argv[3]

    train_set = pd.read_parquet(train_set_path)
    X_train, y_train = df_to_X_y(train_set)
    cv_splits = tscv.split(train_set['date_block_num'].values, n=1, window=16)

    objective = make_xgb_objective(make_xgb_loss(X_train, y_train, cv_splits))

    trials_db = 'sqlite:///%s' % trials_db_path
    study = optuna.create_study(
        direction='minimize', load_if_exists=True, study_name=output_path,
        storage=trials_db, pruner=optuna.pruners.HyperbandPruner())

    n_trials = MAX_EVALS - len(study.trials)
    if n_trials > 0:
        try:
            study.optimize(objective, n_trials=n_trials, n_jobs=1,
                           gc_after_trial=True, catch=(xgb.core.XGBoostError,))
        except KeyboardInterrupt:
            print("Canceling optimization step before it finishes")

    best_ntree_limit = best_num_round(study.best_params, X_train, y_train,
                                      cv_splits)

    reg = sklearn_regressor(study.best_params, best_ntree_limit)

    reg = reg.fit(X_train, y_train)

    print(reg)
    joblib.dump(reg, output_path)
