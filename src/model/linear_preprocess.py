import numpy as np

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

from ..data import get_feature_cols


def make_preprocessor(train_set, test_set):
    features = get_feature_cols(train_set)

    categories = [np.union1d(train_set[feature], test_set[feature]).tolist()
                  for feature in features
                  if feature.startswith('f__cat__')]
    cat_indexes = [i for i, feature in enumerate(features)
                   if feature.startswith('f__cat__')]
    num_indexes = [i for i, feature in enumerate(features)
                   if not feature.startswith('f__cat__')]

    return ColumnTransformer(
        [("onehot", OneHotEncoder(categories=categories), cat_indexes),
         ("scale", RobustScaler(), num_indexes)])
