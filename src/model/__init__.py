import scipy
from sklearn.base import (BaseEstimator, MetaEstimatorMixin, RegressorMixin,
                          TransformerMixin, clone)
import numpy as np


class ClippedOutputRegressor(BaseEstimator, MetaEstimatorMixin,
                             RegressorMixin):
    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, *args, **kwargs):
        self.regressor_ = clone(self.regressor).fit(*args, **kwargs)
        return self

    def predict(self, *args, **kwargs):
        return np.clip(self.regressor_.predict(*args, **kwargs), 0, 20)

    def fit_predict(self, *args, **kwargs):
        self.regressor_ = clone(self.regressor)
        return np.clip(self.regressor_.fit_predict(*args, **kwargs), 0, 20)


# Since I already have the categories I don't need the fit capabilities of the regular onehotencoder
class StaticOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categories, cat_indexes, num_indexes):
        self.categories = categories
        self.cat_indexes = cat_indexes
        self.num_indexes = num_indexes

    def fit(self, *args):
        return self

    def transform(self, X):
        Xts = [scipy.sparse.csr_matrix(X[:, self.num_indexes])]
        for i, col in enumerate(self.cat_indexes):
            Xit = scipy.sparse.csr_matrix(
                (np.ones(X.shape[0]), (np.arange(X.shape[0]), X[:, col])),
                shape=(X.shape[0], np.max(self.categories[i]) + 1))
            Xts.append(Xit)
        return scipy.sparse.hstack(Xts, format='csr')
