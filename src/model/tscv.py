import numpy as np


def split(date_vec, n=3, window=None):
    return [(np.where((date_vec >= (0 if not window else k - window - 1))
                      & (date_vec < k))[0],
             np.where(date_vec == k)[0])
            for k in range(33 - n + 1, 34)]
