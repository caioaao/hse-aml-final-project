import sys

import pandas as pd
from tqdm.auto import tqdm

from . import add_features_deltas

if __name__ == '__main__':
    sales_train = pd.read_parquet(sys.argv[1])
    categories_meta = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    sales_train = sales_train.merge(categories_meta, on='item_id')
    df = sales_train[['item_id', 'shop_id', 'category_name',
                      'date_block_num']].drop_duplicates()

    indexes = [['item_id'],
               ['shop_id'],
               ['category_name'],
               ['item_id', 'shop_id'],
               ['shop_id', 'category_name']]

    for index_cols in tqdm(indexes):
        aux_df = sales_train.groupby(
            index_cols + ['date_block_num'])['item_cnt_day']\
                            .sum().reset_index()
        aux_df.rename(columns={'item_cnt_day': 'item_cnt'}, inplace=True)
        aux_df = add_features_deltas(aux_df, ['item_cnt'],
                                     index_cols=index_cols)
        df = df.merge(aux_df, on=index_cols + ['date_block_num'],
                      how='left', sort=False)
        print(df.shape)

    print(df.columns)
    df.to_parquet(output_path)
