import numpy as np


def split(date_vec, n=3, window=None, test_months=None):
    if test_months is None:
        test_months = range(np.min(date_vec), np.max(date_vec) + 1)

    if n is not None:
        test_months = test_months[-n:]
    return [(np.where((date_vec >= (0 if not window else k - window - 1))
                      & (date_vec < k))[0],
             np.where(date_vec == k)[0])
            for k in test_months]


def train_test_split_indexes(date_vec, train_start=0, train_end=32,
                             test_start=None, test_end=None):
    assert (test_start is None) or (train_end < test_start)
    assert train_start <= train_end
    assert ((test_start is None)
            or (test_end is None)
            or (test_start <= test_end))
    return (np.where((date_vec >= train_start)
                     & (date_vec <= train_end))[0],
            np.where((date_vec >= (train_end + 1 if test_start is None
                                   else test_start)
                      & (date_vec <= test_end if test_end is not None
                         else 33)))[0])


def train_test_split(*arrays, **options):
    train_idx, test_idx = train_test_split_indexes(options.pop('date_vec'),
                                                   **options)
    return [xs[train_idx] for xs in arrays] + [xs[test_idx] for xs in arrays]
