import pandas as pd


if __name__ == '__main__':
    import sys
    sales_train = pd.read_parquet(sys.argv[1])
    output_path = sys.argv[2]

    df = sales_train.groupby(
        by=['date_block_num', 'shop_id', 'item_id'])['item_cnt_day']\
        .sum().reset_index()
    df.rename(columns={'item_cnt_day': 'item_cnt'}, inplace=True)

    df.to_parquet(output_path)
