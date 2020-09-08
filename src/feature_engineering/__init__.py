from tqdm.auto import trange
import pandas as pd

FEATURE_PREFIX = 'f__'
CAT_FEATURE_PREFIX = '%scat__' % FEATURE_PREFIX


def add_lagged_features(
        df, history_df, feature_cols,
        max_lag=32, index_cols=['item_id', 'shop_id'],
        fill_value=0):
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    tmp_df = history_df.copy()
    for k in trange(1, max_lag + 1):
        tmp_df['date_block_num'] = tmp_df['date_block_num'] + 1
        lag_cols = ['%s%s_%d' % (FEATURE_PREFIX, feature_col, k)
                    for feature_col in feature_cols]
        tmp_df[lag_cols] = tmp_df[feature_cols]
        df = df.merge(tmp_df[['date_block_num'] + lag_cols + index_cols],
                      on=index_cols + ['date_block_num'], how='left',
                      sort=False)
    return df.fillna(fill_value)


def drop_non_features(df, inplace=False):
    return df.drop(columns=[c for c in df.columns
                            if not c.startswith(FEATURE_PREFIX)],
                   inplace=inplace)


def get_feature_cols(df):
    return drop_non_features(df).columns


def add_as_features(df, cols, inplace=False, feature_type='numerical'):
    if not inplace:
        df = df.copy()

    if feature_type == 'numerical':
        pref = FEATURE_PREFIX
    elif feature_type == 'categorical':
        pref = CAT_FEATURE_PREFIX

    df[["%s%s" % (pref, col) for col in cols]] = df[cols]

    return df


def add_as_cat_features(df, cols, inplace=False):
    return add_as_features(df, cols, inplace=inplace,
                           feature_type='categorical')


def df_to_X(df):
    return drop_non_features(df).values


def df_to_X_y(df):
    return df_to_X(df), df['item_cnt_month'].values


def _mean_encoding_col_name(on, label):
    return 'mean_%s_on_%s' % (label, '_'.join(on))


def mean_encoding_df(df, on, label='item_cnt'):
    encode_column = _mean_encoding_col_name(on, label)
    return df[on + [label]].groupby(on)[label].mean().reset_index().rename(
        columns={label: encode_column})


def _rolling_mean_encoding_col_name(on, label, w):
    return 'rolling_window_%d_mean_%s_on_%s' % (w, label, '_'.join(on))


def rolling_mean_encoding_df(df, on, label='item_cnt', w=20,
                             date_col='date_block_num'):
    encode_column = _rolling_mean_encoding_col_name(on, label, w)

    dfs = []
    for m in trange(w, 34):
        tmp_df = df.loc[
            (df['date_block_num'] <= m) & (df['date_block_num'] > (m - w)),
            on + [label]].groupby(on)[label].mean().reset_index().rename(
            columns={label: encode_column})
        tmp_df['date_block_num'] = m
        dfs.append(tmp_df)
    return pd.concat(dfs)
