import pandas as pd

from . import add_as_cat_features

if __name__ == '__main__':
    import sys
    input_df = pd.read_parquet(sys.argv[1])
    date_ids = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    df = add_as_cat_features(input_df.merge(date_ids, on='date_block_num',
                                            how='left', sort=False),
                             ['date_block_num', 'month_id', 'year_id'])

    print("%s columns: %s" % (output_path, str(df.columns)))
    df.to_parquet(output_path)
