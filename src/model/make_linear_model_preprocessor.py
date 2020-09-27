import sys

import joblib
import pandas as pd
import numpy as np

from .linear_preprocess import make_linear_preprocessor

from ..data import df_to_X_y
from ..data import get_feature_cols


def _make_preprocessor(train_set, test_set):
    features = get_feature_cols(train_set)

    categories = [np.union1d(train_set[feature], test_set[feature]).tolist()
                  for feature in features
                  if feature.startswith('f__cat__')]
    cat_indexes = [i for i, feature in enumerate(features)
                   if feature.startswith('f__cat__')]
    num_indexes = [i for i, feature in enumerate(features)
                   if not feature.startswith('f__cat__')]

    return make_linear_preprocessor(categories, cat_indexes, num_indexes)


if __name__ == '__main__':
    train_set = pd.read_parquet(sys.argv[1])
    test_set = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    print('Building preprocessor')
    preprocessor = _make_preprocessor(train_set, test_set)
    del test_set

    X, y = df_to_X_y(train_set)
    del train_set

    print('Fitting preprocessor')

    preprocessor.fit(X, y)

    print(preprocessor)
    joblib.dump(preprocessor, output_path)
