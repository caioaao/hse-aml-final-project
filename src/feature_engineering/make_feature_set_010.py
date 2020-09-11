import pandas as pd
import sys

from . import add_lagged_features


if __name__ == '__main__':
    input_df = pd.read_parquet(sys.argv[1])
    ranks_df = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    feature_cols = [col for col in ranks_df if col.endswith('_rank')]

    df = add_lagged_features(input_df, ranks_df, feature_cols,
                             max_lag=3, fill_value=-999)

    print("%s columns: %s" % (output_path, str(df.columns)))
    df.to_parquet(output_path)
