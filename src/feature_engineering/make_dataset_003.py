import pandas as pd

from . import add_lagged_features

if __name__ == '__main__':
    import sys
    input_df = pd.read_parquet(sys.argv[1])
    shop_item_cat_encoding = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    feature_cols = [col for col in shop_item_cat_encoding.columns
                    if col not in ['shop_id', 'date_block_num']]
    df = add_lagged_features(
        input_df, shop_item_cat_encoding, feature_cols, max_lag=1,
        index_cols=['shop_id'])

    df.to_parquet(output_path)
