from tqdm.auto import trange

FEATURE_PREFIX = 'f__'
CAT_FEATURE_PREFIX = '%scat__' % FEATURE_PREFIX


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


def add_as_features(df, cols, inplace=False, feature_type='numerical'):
    if not inplace:
        df = df.copy()

    if feature_type == 'numeric':
        pref = FEATURE_PREFIX
    elif feature_type == 'categorical':
        pref = CAT_FEATURE_PREFIX

    df[["%s%s" % (pref, col) for col in cols]] = df[cols]

    return df


def add_as_cat_features(df, cols, inplace=False):
    return add_as_features(df, cols, inplace=inplace,
                           feature_type='categorical')


def df_to_X_y(df):
    return drop_non_features(df).values, df['item_cnt_month'].values
