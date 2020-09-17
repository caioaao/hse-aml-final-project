import pandas as pd
import numpy as np

from tqdm.auto import trange

FIRST_TRAIN_MONTH = 14

if __name__ == '__main__':
    import sys

    test_set = pd.read_parquet(sys.argv[1])
    sales_by_month = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    test_shops = test_set['shop_id'].unique()
    shops_sales_train = sales_by_month[['shop_id', 'date_block_num']].drop_duplicates()

    train_set = pd.DataFrame(columns=['item_id', 'shop_id', 'date_block_num'])

    for month in trange(FIRST_TRAIN_MONTH, 34):
        candidate_shops = pd.DataFrame({'shop_id': test_shops, 'date_block_num': month - 1})
        candidate_shops.merge(shops_sales_train, on=['shop_id', 'date_block_num'])
        shops = candidate_shops['shop_id']
        items = sales_by_month[sales_by_month['date_block_num'] == month]['item_id'].unique()
        pairs = pd.DataFrame({'date_block_num': month,
                              'item_id': np.repeat(items, shops.shape[0]),
                              'shop_id': np.tile(shops, items.shape[0])})
        train_set = train_set.append(pairs)

    train_set = train_set.merge(
        sales_by_month, on=['item_id', 'shop_id', 'date_block_num'],
        how='left', sort=False).fillna(0)

    train_set.rename(columns={'item_cnt': 'item_cnt_month'}, inplace=True)
    train_set['item_cnt_month'] = np.clip(train_set['item_cnt_month'], 0, 20)
    train_set.to_parquet(output_path)
