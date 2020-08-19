import pandas as pd


if __name__ == '__main__':
    import sys

    sales_train_by_month = pd.read_parquet(sys.argv[1])
    output_path = sys.argv[2]

    sales_train_by_month.loc[
        sales_train_by_month['date_block_num'] > 20,
        ['date_block_num', 'item_id', 'shop_id', 'item_cnt']]\
        .to_parquet(output_path)
