import scipy
import numpy as np
from sklearn.base import (BaseEstimator, MetaEstimatorMixin, TransformerMixin)
from sklearn.preprocessing import StandardScaler


# Since I already have the categories I don't need the fit capabilities of the
# regular onehotencoder
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


# Regular standard scaler was giving OOM errors with larger datasets, so I
# implemented this little batched one
class BatchStandardScaler(BaseEstimator, TransformerMixin,
                          MetaEstimatorMixin):
    def __init__(self, std_scaler, preproc, batch_size=10000):
        self.std_scaler = std_scaler
        self.preproc = preproc
        self.batch_size = batch_size

    def fit(self, X, y=None):
        for i in range(0, X.shape[0], self.batch_size):
            end = min(i + self.batch_size, X.shape[0])
            self.std_scaler.partial_fit(self.preproc.transform(X[i:end, :]))
        return self

    def transform(self, X):
        return self.std_scaler.transform(self.preproc.transform(X))


def make_linear_preprocessor(categories, cat_indexes, num_indexes,
                             batch_size=100000):
    one_hot = StaticOneHotEncoder(categories, cat_indexes, num_indexes)
    scaler = StandardScaler(with_mean=False)
    return BatchStandardScaler(scaler, one_hot, batch_size=batch_size)
