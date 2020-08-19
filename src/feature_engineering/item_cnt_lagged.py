def add_features(df, sales_train, max_lag=32,
                 index_cols=['item_id', 'shop_id'], fill_value=0):
    tmp_df = sales_train.copy()
    for k in range(1, max_lag + 1):
        tmp_df['date_block_num'] = tmp_df['date_block_num'] + 1
        lag_col = 'item_cnt_lagged_%d' % k
        tmp_df[lag_col]
        df = df.merge(tmp_df[['date_block_num', lag_col] + index_cols],
                      on=index_cols + ['date_block_num'], how='left',
                      sort=False)
    return df.fillna(fill_value)
