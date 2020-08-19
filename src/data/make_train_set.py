import pandas as pd
import numpy as np


if __name__ == '__main__':
    import sys

    sales_train_by_month = pd.read_parquet(sys.argv[1])
    output_path = sys.argv[2]

    sales_train_by_month['item_cnt_month'] = \
        np.clip(sales_train_by_month['item_cnt'], 0, 20)

    sales_train_by_month.loc[
        sales_train_by_month['date_block_num'] > 20,
        ['date_block_num', 'item_id', 'shop_id', 'item_cnt_month']]\
        .to_parquet(output_path)
