import pandas as pd

if __name__ == '__main__':
    import sys
    train_set_path = sys.argv[1]
    output_path = sys.argv[2]
    train_set = pd.read_parquet(train_set_path)
    train_set.groupby(['date_block_num', 'shop_id'])['item_cnt']\
             .sum().reset_index()\
             .to_parquet(output_path)
