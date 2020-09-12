import sys
import pandas as pd

from . import add_as_features

if __name__ == '__main__':
    df = pd.read_parquet(sys.argv[1])
    hist_df = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    df = df.merge(hist_df, on=['item_id', 'shop_id', 'date_block_num'],
                  how='left', sort=False).fillna(-999)
    add_as_features(df, [col for col in hist_df
                         if col.endswith('_since_last_seen')],
                    inplace=True)

    print("%s columns: %s" % (output_path, str(df.columns)))
    df.to_parquet(output_path)
