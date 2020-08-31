import pandas as pd
from tqdm.auto import tqdm

from . import (mean_encoding_df, add_lagged_features, rolling_mean_encoding_df)

MEAN_ENCODE_COLS = [["item_id"],
                    ["shop_id"],
                    ["category_name"],
                    ["subcategory_name"],
                    ["item_id", "shop_id"],
                    ["shop_id", "category_name"],
                    ["shop_id", "subcategory_name"]]

if __name__ == '__main__':
    import sys
    df = pd.read_parquet(sys.argv[1])
    sales_train = pd.read_parquet(sys.argv[2])
    categories_metadata = pd.read_parquet(sys.argv[3])
    output_path = sys.argv[4]

    sales_train = sales_train.merge(categories_metadata, on='item_id')

    for cols in tqdm(MEAN_ENCODE_COLS):
        encoding_df = mean_encoding_df(sales_train, cols + ['date_block_num'])
        encoding_col = [col for col in encoding_df.columns
                        if col.startswith('mean_')]
        df = add_lagged_features(
            df, encoding_df, encoding_col,
            max_lag=3, index_cols=cols)

    for cols in tqdm(MEAN_ENCODE_COLS):
        encoding_df = rolling_mean_encoding_df(sales_train, cols)
        encoding_col = [col for col in encoding_df.columns
                        if col.startswith('rolling_window_')]
        df = add_lagged_features(
            df, encoding_df, encoding_col,
            max_lag=1, index_cols=cols)

    df.to_parquet(output_path)
