import pandas as pd

from . import add_lagged_features

if __name__ == '__main__':
    import sys
    input_df = pd.read_parquet(sys.argv[1])
    sales_train_by_month = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    df = add_lagged_features(input_df, sales_train_by_month, 'item_cnt')

    print("%s columns: %s" % (output_path, str(df.columns)))
    df.to_parquet(output_path)
