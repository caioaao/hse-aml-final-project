import pandas as pd
import sys

from . import add_lagged_features

FEATURE_COLS = ['item_shop_price_median',
                'cat_shop_price_median',
                'item_price_median',
                'cat_price_median']


if __name__ == '__main__':
    input_df = pd.read_parquet(sys.argv[1])
    categories_meta = pd.read_parquet(sys.argv[2])
    stats_df = pd.read_parquet(sys.argv[3])
    output_path = sys.argv[4]

    df = input_df.merge(categories_meta, on='item_id')
    df = add_lagged_features(df, stats_df, FEATURE_COLS, fill_value=-999,
                             max_lag=6)

    print("%s columns: %s" % (output_path, str(df.columns)))
    df.to_parquet(output_path)
