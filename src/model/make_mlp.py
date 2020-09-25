import sys

import joblib
import pandas as pd

from ..data import df_to_X_y
from ..functional import pipe
from .mlp import MLPRegressor


if __name__ == '__main__':
    train_set = pd.read_parquet(sys.argv[1])
    preprocessor = joblib.load(sys.argv[2])
    output_path = sys.argv[3]

    print('Loading dataset')
    X, y = df_to_X_y(train_set)
    del train_set

    print('Building final estimator')

    mlp = MLPRegressor(preprocessor=preprocessor, batch_size=1024)

    print('Fitting final estimator')
    mlp.fit(X, y)
    joblib.dump(mlp, output_path)
