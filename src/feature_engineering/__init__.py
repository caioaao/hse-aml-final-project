from tqdm.auto import trange


FEATURE_PREFIX = 'f__'


def add_lagged_features(
        df, sales_train, feature_col,
        max_lag=32, index_cols=['item_id', 'shop_id'],
        fill_value=0):
    tmp_df = sales_train.copy()
    for k in trange(1, max_lag + 1):
        tmp_df['date_block_num'] = tmp_df['date_block_num'] + 1
        lag_col = '%s%s_%d' % (FEATURE_PREFIX, feature_col, k)
        tmp_df[lag_col] = tmp_df[feature_col]
        df = df.merge(tmp_df[['date_block_num', lag_col] + index_cols],
                      on=index_cols + ['date_block_num'], how='left',
                      sort=False)
    return df.fillna(fill_value)


def drop_non_features(df, inplace=False):
    return df.drop(columns=[c for c in df.columns
                            if not c.startswith(FEATURE_PREFIX)],
                   inplace=inplace)
