import sys
import pandas as pd
from tqdm.auto import tqdm

if __name__ == '__main__':
    sales_train = pd.read_parquet(sys.argv[1])
    categories_meta = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    sales_train = sales_train.merge(categories_meta, on='item_id')

    ranks_index_cols = [['item_id'],
                        ['category_name'],
                        ['shop_id', 'category_name']]

    agg = sales_train.groupby(['item_id', 'shop_id', 'category_name',
                               'date_block_num'])\
                     .agg({'item_cnt_day': 'sum',
                           'item_price': 'median'})\
                     .reset_index()

    ranks_df = agg[['item_id', 'shop_id', 'category_name',
                    'date_block_num']].copy()

    for idx_cols in tqdm(ranks_index_cols):
        price_rank_col = '%s_item_price_median_rank' % ('_'.join(idx_cols))
        item_cnt_rank_col = '%s_item_cnt_month_rank' % ('_'.join(idx_cols))

        grp = agg.groupby(idx_cols + ['date_block_num'])

        ranks_df[price_rank_col] = grp['item_price'].rank(method='dense',
                                                          pct=True)
        ranks_df[item_cnt_rank_col] = grp['item_cnt_day'].rank(method='dense',
                                                               pct=True)

    ranks_df.to_parquet(output_path)
