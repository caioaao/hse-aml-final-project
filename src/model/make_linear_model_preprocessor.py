import sys

import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from . import StaticOneHotEncoder

from ..data import df_to_X_y
from ..data import get_feature_cols


def make_onehot(train_set, test_set):
    features = get_feature_cols(train_set)

    categories = [np.union1d(train_set[feature], test_set[feature]).tolist()
                  for feature in features
                  if feature.startswith('f__cat__')]
    cat_indexes = [i for i, feature in enumerate(features)
                   if feature.startswith('f__cat__')]
    num_indexes = [i for i, feature in enumerate(features)
                   if not feature.startswith('f__cat__')]

    return StaticOneHotEncoder(categories, cat_indexes, num_indexes)


# fitting in batch to stop memory issues
def fit_std_scaler(t, onehot, X):
    batch_size = 10000
    for i in tqdm(range(0, X.shape[0], batch_size)):
        end = min(i + batch_size, X.shape[0])
        t.partial_fit(onehot.transform(X[i:end, :]))


if __name__ == '__main__':
    train_set = pd.read_parquet(sys.argv[1])
    test_set = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    X, y = df_to_X_y(train_set)

    print('Building preprocessor')
    one_hot = make_onehot(train_set, test_set)
    std_scaler = StandardScaler(with_mean=False)
    del test_set
    del train_set

    print('Fitting one-hot encoder')
    one_hot.fit(X)

    print('Fitting standard scaler')

    fit_std_scaler(std_scaler, one_hot, X)

    preprocessor = Pipeline(
        [('onehot', one_hot),
         ('scale', std_scaler)])

    print(preprocessor)
    joblib.dump(preprocessor, output_path)
