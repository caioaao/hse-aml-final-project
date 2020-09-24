import sys
import pandas as pd
import optuna
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from src.data import df_to_X_y
from src.model import tscv


MAX_EVALS = 100


def rf_from_trial(trial: optuna.Trial):
    max_leaf_nodes = trial.suggest_categorical(
        'max_leaf_nodes_type', ['unlimited', 'limited'])
    if max_leaf_nodes == 'unlimited':
        max_leaf_nodes = None
    else:
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 1, 1000)
    params = {'n_estimators': trial.suggest_int('n_estimators', 10, 300),
              'criterion': trial.suggest_categorical(
                  'criterion', ['mse', 'mae']),
              'min_samples_split': trial.suggest_uniform('min_samples_split',
                                                         0., 1.),
              'min_samples_leaf': trial.suggest_uniform('min_samples_leaf',
                                                        0., .5),
              'max_features': trial.suggest_categorical(
                  'max_features', ['auto', 'log2', 'sqrt', None]),
              'max_leaf_nodes': max_leaf_nodes,
              'random_state': trial.suggest_int('random_state', 0, 999999),
              'n_jobs': -1,
              'verbose': 1}
    return RandomForestRegressor(**params)


def make_objective(Xtrain, ytrain, cv_splits):
    def objective(trial):
        return np.mean(cross_val_score(
            rf_from_trial(trial), Xtrain, ytrain, cv=cv_splits,
            scoring='neg_root_mean_squared_error'))
    return objective


if __name__ == '__main__':
    train_set = pd.read_parquet(sys.argv[1])
    trials_db_path = sys.argv[2]
    output_path = sys.argv[3]

    trials_db = 'sqlite:///%s' % trials_db_path
    study = optuna.create_study(
        direction='maximize', load_if_exists=True, study_name=output_path,
        storage=trials_db)

    Xtrain, ytrain = df_to_X_y(train_set)
    cv_splits = tscv.split(train_set['date_block_num'], window=16, n=1)
    del train_set

    n_trials = MAX_EVALS - len(study.trials)
    if n_trials > 0:
        study.optimize(make_objective(Xtrain, ytrain, cv_splits),
                       n_trials=n_trials, n_jobs=1,
                       gc_after_trial=True)
