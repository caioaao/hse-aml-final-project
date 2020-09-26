import joblib
import pandas as pd
from tqdm.auto import tqdm

from ..data import df_to_X_y


from . import tscv

if __name__ == '__main__':
    import sys
    reg_path = sys.argv[1]
    train_set = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    num_months = len(train_set['date_block_num'].unique())

    if num_months < 10:
        n_splits = 1
        window = num_months - 1
    else:
        n_splits = 8
        window = min(16, num_months - 1)

    cv_splits = tscv.split(train_set['date_block_num'], n=n_splits,
                           window=window)
    reg = joblib.load(reg_path)

    X, y = df_to_X_y(train_set)
    del train_set

    output_dfs = []
    for fold, (train, test) in enumerate(tqdm(cv_splits)):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        reg.fit(X_train, y_train)
        output_dfs.append(pd.DataFrame({'oof_preds': reg.predict(X_test),
                                        'oof_idx': test,
                                        'item_cnt_month': y[test],
                                        'fold_id': fold}))

    pd.concat(output_dfs, axis=0).to_parquet(output_path)
