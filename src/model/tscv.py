import numpy as np


def split(X, n=3, date_col=0, window=None):
    dates = X[:, date_col]
    return [(np.where((dates >= (0 if not window else k - window - 1))
                      & (dates < k))[0],
             np.where(dates == k)[0])
            for k in range(33 - n + 1, 34)]
