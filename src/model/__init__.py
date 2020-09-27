from sklearn.base import (BaseEstimator, MetaEstimatorMixin, RegressorMixin,
                          clone)
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
