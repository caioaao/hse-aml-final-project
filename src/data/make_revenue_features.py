import sys

import pandas as pd

INDEX_COLS = [['item_id'],
              ['shop_id'],
              ['category_name'],
              ['item_id', 'shop_id'],
              ['category_name', 'shop_id']]

if __name__ == '__main__':
    sales_train = pd.read_parquet(sys.argv[1])
    categories_meta = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    sales_train = sales_train.merge(categories_meta, on='item_id')

    hist_df = sales_train[['item_id', 'shop_id', 'category_name',
                           'date_block_num']].drop_duplicates()

    for cols in INDEX_COLS:
        col_id = '_'.join(cols)
        item_cnt_col = '%s_item_cnt' % col_id
        item_price_col = '%s_avg_item_price' % col_id
        sales_revenue_col = '%s_sales_revenue' % col_id
        sales_price_ratio_col = '%s_sales_price_ratio' % col_id

        grp = sales_train.groupby(cols + ['date_block_num'])
        sales_df = grp['item_cnt_day'].sum().reset_index()
        sales_df.rename(columns={'item_cnt_day': item_cnt_col}, inplace=True)

        prices_df = grp['item_price'].mean().reset_index()
        prices_df.rename(columns={'item_price': item_price_col}, inplace=True)

        aux_df = pd.merge(sales_df, prices_df, on=cols + ['date_block_num'])
        aux_df[sales_revenue_col] = aux_df[item_cnt_col]\
            * aux_df[item_price_col]
        aux_df[sales_price_ratio_col] = aux_df[item_cnt_col]\
            / aux_df[item_price_col]

        hist_df = hist_df.merge(aux_df, on=cols + ['date_block_num'],
                                how='left', sort=False)

    hist_df.to_parquet(output_path)
