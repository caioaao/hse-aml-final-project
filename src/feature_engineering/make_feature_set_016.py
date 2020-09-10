import sys
import pandas as pd

from . import add_lagged_features

if __name__ == '__main__':
    df = pd.read_parquet(sys.argv[1])
    sales_deltas = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    features_cols = [col for col in sales_deltas.columns
                     if '_delta_' in col]
    df = add_lagged_features(df, sales_deltas, features_cols, max_lag=1,
                             fill_value=0)

    print("%s columns: %s" % (output_path, str(df.columns)))
    df.to_parquet(output_path)
