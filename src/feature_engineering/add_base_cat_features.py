import pandas as pd


def _add_features(df):
    df[['cat__item_id', 'cat__shop_id', 'cat__date_block_num']] = df[['item_id', 'shop_id', 'date_block_num']]


if __name__ == '__main__':
    import sys
    input_df = pd.read_parquet(sys.argv[1])
    output_path = sys.argv[2]

    _add_features(input_df)

    input_df.to_parquet(output_path)
