import sys

import pandas as pd


RELEASE_DATE_GROUPS = [['item_id'],
                       ['item_id', 'shop_id'],
                       ['shop_id']]


if __name__ == '__main__':
    sales_train = pd.read_parquet(sys.argv[1])
    output_path = sys.argv[2]

    df = sales_train[['item_id', 'shop_id']].drop_duplicates()

    for group in RELEASE_DATE_GROUPS:
        group_id = '_'.join(group)
        release_date_col = '%s_release_date' % group_id
        release_dates = sales_train.groupby(group)['date_block_num']\
                                   .min()\
                                   .reset_index()
        release_dates.rename(columns={'date_block_num': release_date_col},
                             inplace=True)
        df = df.merge(release_dates, on=group, how='left').fillna(999)

    print(df.columns)
    df.to_parquet(output_path)
