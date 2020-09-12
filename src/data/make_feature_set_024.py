import sys

import pandas as pd
from tqdm.auto import tqdm

from . import add_as_features


GROUPS = [['item_id'],
          ['item_id', 'shop_id'],
          ['shop_id']]


if __name__ == '__main__':
    df = pd.read_parquet(sys.argv[1])
    sales_train = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    months_in_df = df['date_block_num'].unique().tolist()
    for group in tqdm(GROUPS):
        group_id = '_'.join(group)
        date_col = '%s_last_seen_date' % group_id
        delta_col = '%s_since_last_seen' % group_id
        months_dfs = []
        for month in tqdm(months_in_df):
            month_df = sales_train[sales_train['date_block_num'] < month]\
                .groupby(group)['date_block_num'].max().reset_index()
            month_df.rename(columns={'date_block_num': date_col}, inplace=True)
            month_df['date_block_num'] = month
            months_dfs.append(month_df)
        grp_df = pd.concat(months_dfs, axis=0)
        df = df.merge(grp_df, on=group + ['date_block_num'], how='left')
        df[delta_col] = df['date_block_num'] - df[date_col]
    df.fillna(-999, inplace=True)

    add_as_features(df, [col for col in df
                         if col.endswith('_since_last_seen')],
                    inplace=True)

    df.to_parquet(output_path)
