import sys
import pandas as pd

from . import add_as_features

if __name__ == '__main__':
    df = pd.read_parquet(sys.argv[1])
    prices_deltas = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    prices_deltas['date_block_num'] = prices_deltas['date_block_num'] + 1

    df = df.merge(prices_deltas, on=['item_id', 'shop_id', 'date_block_num'],
                  how='left', sort=False)
    df.fillna(0, inplace=True)

    add_as_features(df, [col for col in prices_deltas.columns
                         if '_delta_' in col],
                    inplace=True)

    print("%s columns: %s" % (output_path, str(df.columns)))
    df.to_parquet(output_path)
