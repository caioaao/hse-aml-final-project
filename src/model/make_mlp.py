import sys

import joblib
import pandas as pd

from ..data import df_to_X_y
from .mlp import MLPRegressor
from . import tscv

from .early_stopping import early_stopping_fit


def optimize_n_epochs(train_set_path, preprocessor):
    print('Loading dataset')
    train_set = pd.read_parquet(train_set_path)
    X_train, y_train, X_val, y_val = tscv.train_test_split(
        *df_to_X_y(train_set),
        date_vec=train_set['date_block_num'].values,
        train_start=16)
    del train_set

    print('Finding optimal number of epochs')
    _, n_epochs = early_stopping_fit(MLPRegressor(preprocessor=preprocessor,
                                                  batch_size=128000),
                                     X_train, y_train, X_val, y_val,
                                     max_iter=50)

    print(f'Best n_epochs={n_epochs}')

    return n_epochs


if __name__ == '__main__':
    train_set_path = sys.argv[1]
    preprocessor = joblib.load(sys.argv[2])
    output_path = sys.argv[3]

    n_epochs = optimize_n_epochs(train_set_path, preprocessor)

    print('Building final estimator')
    mlp = MLPRegressor(preprocessor=preprocessor, n_epochs=n_epochs,
                       batch_size=128000)

    print('Loading dataset')
    X, y = df_to_X_y(pd.read_parquet(train_set_path), window=16)

    print('Fitting final estimator')
    mlp.fit(X, y)
    joblib.dump(mlp, output_path)
