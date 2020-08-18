import pandas as pd
from tqdm.auto import trange

if __name__ == '__main__':
    import sys
    train_set = pd.read_parquet(sys.argv[1])
    test_set = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    df_lags = pd.concat(
        [train_set[['item_id', 'shop_id', 'date_block_num']],
         test_set[['item_id', 'shop_id', 'date_block_num']]], axis=0)
    for lag in trange(1, 33):
        train_set_lagged = \
            train_set.rename(columns={'item_cnt': 'item_cnt_lag_%d' % lag})
        train_set_lagged['date_block_num'] = \
            train_set_lagged['date_block_num'] + lag
        df_lags = df_lags.merge(train_set_lagged,
                                on=['item_id', 'shop_id', 'date_block_num'],
                                how='left')
    df_lags.fillna(0, inplace=True)
    df_lags.to_parquet(output_path)
