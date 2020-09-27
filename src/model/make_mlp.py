import gc
import sys

import joblib
import pandas as pd
from sklearn.pipeline import make_pipeline
import scipy

from ..data import df_to_X_y, df_to_X
from .mlp import MLPRegressor
from . import tscv

from .early_stopping import early_stopping_fit


BATCH_SIZE = 1024


def optimize_n_epochs(train_set_path, preprocessor):
    print('Loading dataset')
    train_set = pd.read_parquet(train_set_path)
    X_train, y_train, X_val, y_val = tscv.train_test_split(
        *df_to_X_y(train_set),
        date_vec=train_set['date_block_num'].values,
        train_start=16)
    del train_set
    X_train = preprocessor.transform(X_train)
    X_val = preprocessor.transform(X_val)

    print('Finding optimal number of epochs')
    _, n_epochs = early_stopping_fit(MLPRegressor(batch_size=BATCH_SIZE),
                                     X_train, y_train, X_val, y_val,
                                     max_iter=50)

    print(f'Best n_epochs={n_epochs}')

    return n_epochs


if __name__ == '__main__':
    train_set_path = sys.argv[1]
    preprocessor = joblib.load(sys.argv[2])
    output_path = sys.argv[3]

    n_epochs = optimize_n_epochs(train_set_path, preprocessor)
    gc.collect()

    print('Loading dataset again')
    train_set = pd.read_parquet(train_set_path)
    X, y = df_to_X_y(train_set, window=16)
    del train_set
    gc.collect()
    print('Transforming X before fit')
    X = scipy.sparse.vstack([preprocessor.transform(X[:1000000, :]),
                             preprocessor.transform(X[1000000:, :])])

    print('Building final estimator')
    mlp = MLPRegressor(n_epochs=n_epochs, batch_size=BATCH_SIZE)

    print('Fitting final estimator')
    mlp.fit(X, y)
    reg = make_pipeline(preprocessor, mlp)
    joblib.dump(reg, output_path)
