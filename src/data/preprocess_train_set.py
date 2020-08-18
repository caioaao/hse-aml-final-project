import pandas as pd


if __name__ == '__main__':
    import sys

    sales_train_path = sys.argv[1]
    output_path = sys.argv[2]

    sales_train = pd.read_parquet(sales_train_path)
    sales_train[['date_block_num', 'item_id', 'shop_id', 'item_cnt']].to_parquet(output_path)
