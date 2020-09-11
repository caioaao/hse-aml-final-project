import sys
import pandas as pd

if __name__ == '__main__':
    sales_train = pd.read_parquet(sys.argv[1])
    categories_meta = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    sales_train = sales_train.merge(categories_meta, on='item_id')
    df = sales_train[['item_id', 'shop_id', 'category_name',
                      'date_block_num']].copy()
    df.drop_duplicates(inplace=True)

    stats_confs = [['item_id', 'shop_id'],
                   ['category_name'],
                   ['category_name', 'shop_id'],
                   ['item_id']]

    for cols in stats_confs:
        idx = cols + ['date_block_num']
        feature_col = '%s_price_median' % '_'.join(cols)

        stat_df = sales_train.groupby(idx)['item_price']\
                             .median().reset_index()\
                             .rename(columns={'item_price': feature_col})
        df = df.merge(stat_df, on=idx, how='outer', sort=False)

    idxs_to_imput = [['item_id', 'shop_id'],
                     ['category_name', 'shop_id'],
                     ['item_id']]
    imput_src = 'category_name_price_median'
    for idxs in idxs_to_imput:
        col = '%s_price_median' % '_'.join(idxs)
        df.loc[df[col].isnull(), col] = df.loc[df[col].isnull(),
                                               imput_src]
    df = df[['item_id', 'shop_id', 'date_block_num'] +
            [col for col in df.columns if col.endswith('_median')]]

    print("%s columns: %s" % (output_path, str(df.columns)))
    print("%s nulls: %s" % (output_path, str(df.isnull().sum() / df.shape[0])))
    df.to_parquet(output_path)
