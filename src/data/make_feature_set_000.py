import pandas as pd

from . import add_as_cat_features

if __name__ == '__main__':
    import sys
    input_df = pd.read_parquet(sys.argv[1])
    output_path = sys.argv[2]

    # dropped item_id because it'd break linear models when trying to one-hot
    # encode them and it actually resulted in better results
    df = add_as_cat_features(input_df, ['shop_id', 'date_block_num'])
    print("%s columns: %s" % (output_path, str(df.columns)))
    df.to_parquet(output_path)
