import pandas as pd
import numpy as np

if __name__ == '__main__':
    import sys
    train_set = pd.read_parquet(sys.argv[1])
    item_categories_metadata = pd.read_parquet(sys.argv[2])
    shop_sales_train_by_month = pd.read_parquet(sys.argv[3])
    output_path = sys.argv[4]

    train_set_with_categories_metadata = train_set.merge(
        item_categories_metadata, on='item_id')

    shop_item_cat_encoding = train_set_with_categories_metadata\
        .pivot_table(index=['shop_id', 'date_block_num'],
                     columns='category_name',
                     aggfunc=np.sum, fill_value=0,
                     values='item_cnt').reset_index()
    shop_item_cat_encoding.columns.name = None
    category_names = \
        item_categories_metadata['category_name'].drop_duplicates().values
    category_names.sort()
    encoding_cols = ['shop_item_cat_enc_%d' % i
                     for i, cat_name in enumerate(category_names)]
    shop_item_cat_encoding.rename(
        columns=dict(zip(category_names, encoding_cols)),
        inplace=True)
    shop_item_cat_encoding = shop_item_cat_encoding.merge(
        shop_sales_train_by_month, how='left',
        on=['date_block_num', 'shop_id'])
    shop_item_cat_encoding[encoding_cols] = \
        shop_item_cat_encoding[encoding_cols].div(
            shop_item_cat_encoding['item_cnt'], axis=0)
    shop_item_cat_encoding.drop(columns='item_cnt', inplace=True)
    shop_item_cat_encoding.to_parquet(output_path)
