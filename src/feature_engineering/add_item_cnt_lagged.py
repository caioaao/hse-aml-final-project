import pandas as pd


if __name__ == '__main__':
    import sys
    input_df = pd.read_parquet(sys.argv[1])
    item_count_lagged = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    item_count_lagged.drop(columns=['item_cnt_lag_%d' % d
                                    for d in range(19, 33)],
                           inplace=True)

    df = input_df.merge(
        item_count_lagged, on=['item_id', 'shop_id', 'date_block_num'],
        how='left', sort=False)

    df.to_parquet(output_path)
