import sys
import pandas as pd

from . import add_lagged_features

if __name__ == '__main__':
    df = pd.read_parquet(sys.argv[1])
    hist_df = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    df = add_lagged_features(
        df, hist_df, [col for col in hist_df.columns
                      if col.endswith('_revenue') or col.endswith('ratio')],
        max_lag=3)

    print("%s columns: %s" % (output_path, str(df.columns)))
    df.to_parquet(output_path)
