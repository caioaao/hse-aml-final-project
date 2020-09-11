import pandas as pd
import sys

from . import add_lagged_features

if __name__ == '__main__':
    input_df = pd.read_parquet(sys.argv[1])
    categories_meta = pd.read_parquet(sys.argv[2])
    stats_df = pd.read_parquet(sys.argv[3])
    output_path = sys.argv[4]

    feature_cols = [col for col in stats_df.columns if col.endswith('_median')]
    df = input_df.merge(categories_meta, on='item_id')
    df = add_lagged_features(df, stats_df, feature_cols, fill_value=0,
                             max_lag=3)

    print("%s columns: %s" % (output_path, str(df.columns)))
    df.to_parquet(output_path)
