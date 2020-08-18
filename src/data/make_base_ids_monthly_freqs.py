import pandas as pd
if __name__ == '__main__':
    import sys

    train_set = pd.read_parquet(sys.argv[1])
    item_categories_metadata = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    train_set.merge(item_categories_metadata, on='item_id')\
             .groupby('date_block_num')\
             .agg({'category_name': 'nunique',
                   'shop_id': 'nunique',
                   'item_id': 'nunique'})\
             .reset_index()\
             .rename(columns={'category_name': 'nunique_category_names',
                              'shop_id': 'nunique_shop_ids',
                              'item_id': 'nunique_item_ids'})\
             .to_parquet(output_path)
