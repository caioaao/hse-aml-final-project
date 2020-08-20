import numpy as np


def split(date_vec, n=3, window=None, test_months=None):
    test_months = test_months if test_months is not None else range(1, 34)
    if n is not None:
        test_months = test_months[-n:]
    return [(np.where((date_vec >= (0 if not window else k - window - 1))
                      & (date_vec < k))[0],
             np.where(date_vec == k)[0])
            for k in test_months]
