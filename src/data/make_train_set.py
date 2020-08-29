import pandas as pd
import numpy as np


if __name__ == '__main__':
    import sys

    test_set = pd.read_parquet(sys.argv[1])
    sales_by_month = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    def make_train_set_month(m):
        df = test_set[['item_id', 'shop_id']].copy()
        df['date_block_num'] = m
        return df

    train_set = pd.concat([make_train_set_month(m) for m in range(10, 34)])
    train_set = train_set.merge(
        sales_by_month, on=['item_id', 'shop_id', 'date_block_num'],
        how='left', sort=False).fillna(0)

    train_set.rename(columns={'item_cnt': 'item_cnt_month'}, inplace=True)
    train_set['item_cnt_month'] = np.clip(train_set['item_cnt_month'], 0, 20)
    train_set.to_parquet(output_path)
