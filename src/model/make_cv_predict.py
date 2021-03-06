import joblib
import pandas as pd
from tqdm.auto import tqdm

from ..data import df_to_X_y

from . import tscv


def _get_batch(idx, train_set_path):
    train_set = pd.read_parquet(train_set_path)
    X, y = df_to_X_y(train_set)
    return X[idx], y[idx]


if __name__ == '__main__':
    import sys
    reg_path = sys.argv[1]
    train_set_path = sys.argv[2]
    output_path = sys.argv[3]

    print('Loading data for calculating CV splits')
    train_set = pd.read_parquet(train_set_path)
    num_months = len(train_set['date_block_num'].unique())

    if num_months < 10:
        n_splits = 1
        window = num_months - 1
    else:
        n_splits = 8
        window = min(16, num_months - 1)

    cv_splits = tscv.split(train_set['date_block_num'], n=n_splits,
                           window=window)
    del train_set

    print('Starting CV predict')
    reg = joblib.load(reg_path)
    output_dfs = []
    for fold, (train, test) in enumerate(tqdm(cv_splits)):
        X_train, y_train = _get_batch(train, train_set_path)
        reg.fit(X_train, y_train)
        del X_train
        del y_train
        X_test, y_test = _get_batch(test, train_set_path)
        output_dfs.append(pd.DataFrame({'oof_preds': reg.predict(X_test),
                                        'oof_idx': test,
                                        'item_cnt_month': y_test,
                                        'fold_id': fold}))
        del X_test
        del y_test

    pd.concat(output_dfs, axis=0).to_parquet(output_path)
