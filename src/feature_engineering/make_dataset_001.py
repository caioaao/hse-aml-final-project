import pandas as pd

from . import add_lagged_features, add_as_cat_features

if __name__ == '__main__':
    import sys
    input_df = pd.read_parquet(sys.argv[1])
    sales_train_by_month = pd.read_parquet(sys.argv[2])
    date_ids = pd.read_parquet(sys.argv[3])
    output_path = sys.argv[4]

    df = add_as_cat_features(input_df.merge(date_ids, on='date_block_num',
                                            how='left', sort=False),
                             ['date_block_num', 'month_id', 'year_id'])
    df = add_lagged_features(df, sales_train_by_month, 'item_cnt')

    df.to_parquet(output_path)
