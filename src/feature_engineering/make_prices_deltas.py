import sys

import pandas as pd
from tqdm.auto import tqdm

from ..feature_engineering import add_features_deltas

if __name__ == '__main__':
    prices_stats = pd.read_parquet(sys.argv[1])
    output_path = sys.argv[2]

    df = add_features_deltas(prices_stats, [col for col in prices_stats.columns
                                            if col.endswith('_median')])
    print(df.shape)

    print(df.columns)
    df.to_parquet(output_path)
