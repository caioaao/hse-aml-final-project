import sys
import pandas as pd

from . import add_as_features

if __name__ == '__main__':
    df = pd.read_parquet(sys.argv[1])
    release_dates = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    df = df.merge(release_dates, on=['item_id', 'shop_id'],
                  how='left', sort=False).fillna(999)
    dates_cols = [col for col in release_dates.columns
                  if col.endswith('_release_date')]

    for col in dates_cols:
        feature_col = col.replace('_release_date', '_months_since_launch')
        df[feature_col] = df['date_block_num'] - df[col]
        df.loc[df[feature_col] <= 0, feature_col] = -999

    df.drop(columns=dates_cols, inplace=True)  # to avoid a data leak

    add_as_features(df, [col for col in df
                         if col.endswith('_since_launch')],
                    inplace=True)

    print("%s columns: %s" % (output_path, str(df.columns)))
    df.to_parquet(output_path)
