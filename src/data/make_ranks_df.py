import sys
import pandas as pd
from tqdm.auto import tqdm

if __name__ == '__main__':
    prices_stats = pd.read_parquet(sys.argv[1])
    train_set = pd.read_parquet(sys.argv[2])
    categories_meta = pd.read_parquet(sys.argv[3])
    output_path = sys.argv[4]

    prices_stats = prices_stats.merge(categories_meta, on='item_id')
    train_set = train_set.merge(categories_meta, on='item_id')

    ranks_index_cols = [['item_id'],
                        ['category_name'],
                        ['shop_id', 'category_name']]

    ranks_df = train_set.copy()
    for idx_cols in tqdm(ranks_index_cols):
        price_rank_col = 'rank_%s_item_price_median' % ('_'.join(idx_cols))
        item_cnt_rank_col = 'rank_%s_item_cnt_month' % ('_'.join(idx_cols))

        ranks_df[price_rank_col] = prices_stats.groupby(
            idx_cols + ['date_block_num'])['item_price_median'].rank(
                method='dense', pct=True)
        ranks_df[item_cnt_rank_col] = train_set.groupby(
            idx_cols + ['date_block_num'])['item_cnt_month'].rank(
                method='dense', pct=True)

    ranks_df.to_parquet(output_path)
