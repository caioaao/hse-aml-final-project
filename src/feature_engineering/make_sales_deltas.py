import sys

import pandas as pd
from tqdm.auto import tqdm

from ..feature_engineering import add_features_deltas

if __name__ == '__main__':
    df = pd.read_parquet(sys.argv[1])
    sales_by_month = pd.read_parquet(sys.argv[2])
    categories_meta = pd.read_parquet(sys.argv[3])
    output_path = sys.argv[4]

    sales_by_month = sales_by_month.merge(categories_meta, on='item_id')
    df = df.merge(categories_meta, on='item_id')

    indexes = [['item_id'],
               ['shop_id'],
               ['category_name'],
               ['item_id', 'shop_id'],
               ['shop_id', 'category_name']]

    for index_cols in tqdm(indexes):
        aux_df = sales_by_month.groupby(
            index_cols + ['date_block_num'])['item_cnt'].sum().reset_index()
        aux_df = add_features_deltas(aux_df, ['item_cnt'],
                                     index_cols=index_cols)
        df = df.merge(aux_df, on=index_cols + ['date_block_num'],
                      how='left', sort=False)
        print(df.shape)

    print(df.columns)
    df.to_parquet(output_path)
