import xgboost as xgb
from xgboost import XGBRegressor
from src.model.metrics import corrected_rmse
from optuna import Trial


DEFAULT_PARAMS = {"objective": "reg:squarederror",
                  "n_jobs": -1,
                  "base_score": 0.5,
                  "scale_pos_weight": 1}


def _xgb_feval(y_pred, dtrain):
    return 'cRMSE', corrected_rmse(dtrain.get_label(), y_pred)


def make_xgb_loss(X_train, y_train, cv_splits, verbose=True):
    dtrain = xgb.DMatrix(X_train, y_train)
    return lambda params: xgb.cv(
        params, dtrain, folds=cv_splits, verbose_eval=verbose,
        feval=_xgb_feval, maximize=False)['test-cRMSE-mean'].min()


def trial_to_params(trial: Trial):
    return {**DEFAULT_PARAMS,
            "max_depth": trial.suggest_int('max_depth', 2, 20, 1),
            "subsample": trial.suggest_discrete_uniform('subsample',
                                                        .20, 1.00, .01),
            "colsample_bytree": trial.suggest_discrete_uniform(
                'colsample_bytree', .20, 1., .01),
            "colsample_bylevel": trial.suggest_discrete_uniform(
                'colsample_bylevel', .20, 1., .01),
            "seed": trial.suggest_int('seed', 0, 999999),
            "learning_rate": trial.suggest_uniform(
                'learning_rate', 0.01, 0.15),
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
            "reg_alpha": trial.suggest_categorical("reg_alpha", [0, 0, 0, 0, 0,
                                                                 0.00000001,
                                                                 0.00000005,
                                                                 0.0000005,
                                                                 0.000005]),
            "reg_lambda": trial.suggest_categorical("reg_lambda", [1, 1, 1, 1,
                                                                   2, 3, 4, 5,
                                                                   1])}


def make_xgb_objective(xgb_loss):
    return lambda trial: xgb_loss(trial_to_params(trial))


def train(params, X_train, y_train):
    dtrain = xgb.DMatrix(X_train, y_train)
    return xgb.train({**DEFAULT_PARAMS, **params}, dtrain)


def best_num_round(params, X, y, cv_splits):
    params = {**DEFAULT_PARAMS, **params}
    train_idx, test_idx = cv_splits[-1]
    dtrain = xgb.DMatrix(X[train_idx], y[train_idx])
    dtest = xgb.DMatrix(X[test_idx], y[test_idx])
    bst = xgb.train(params, dtrain, early_stopping_rounds=50,
                    num_boost_round=1000, evals=[(dtrain, 'dtrain'),
                                                 (dtest, 'dtest')],
                    feval=_xgb_feval)
    return bst.best_ntree_limit


def sklearn_regressor(params, num_round):
    return XGBRegressor(n_estimators=num_round, **DEFAULT_PARAMS, **params)
