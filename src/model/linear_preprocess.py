import numpy as np

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from ..data import get_feature_cols


def make_onehot_encoder(train_set, test_set):
    features = get_feature_cols(train_set)

    categories = []
    indexes = []

    for i, feature in enumerate(features):
        if feature.startswith('f__cat__'):
            cats = np.union1d(train_set[feature], test_set[feature]).tolist()
            categories.append(cats)
            indexes.append(i)

    return ColumnTransformer(
        [("onehot", OneHotEncoder(categories=categories), indexes)],
        remainder='passthrough')


def make_preprocessor(train_set, test_set):
    return make_pipeline(make_onehot_encoder(train_set, test_set),
                         RobustScaler())
