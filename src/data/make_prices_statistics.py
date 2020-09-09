import sys
import pandas as pd
import zipfile

if __name__ == '__main__':
    raw_data_path = sys.argv[1]
    train_set = pd.read_parquet(sys.argv[2])
    categories_meta = pd.read_parquet(sys.argv[3])
    output_path = sys.argv[4]

    with zipfile.ZipFile(raw_data_path, 'r') as datasets_file:
        sales_train = pd.read_csv(datasets_file.open('sales_train.csv'))

    df = train_set.merge(categories_meta, on='item_id')

    sales_train = sales_train[(sales_train['item_price'] >= 0)
                              & (sales_train['item_price'] < 3e5)]
    sales_train = sales_train.merge(categories_meta, on='item_id')

    stats_confs = [{'col': 'item_shop_price_median',
                    'index_cols': ['item_id', 'shop_id']},
                   {'col': 'cat_price_median',
                    'index_cols': ['category_name']},
                   {'col': 'cat_shop_price_median',
                    'index_cols': ['category_name', 'shop_id']},
                   {'col': 'item_price_median',
                    'index_cols': ['item_id']}]

    for conf in stats_confs:
        idx = conf['index_cols'] + ['date_block_num']
        stat_df = sales_train.groupby(idx)['item_price']\
                             .median().reset_index()\
                             .rename(columns={'item_price': conf['col']})
        df = df.merge(stat_df, on=idx, how='left', sort=False)

    cols_to_imput = ['item_shop_price_median',
                     'cat_shop_price_median',
                     'item_price_median']
    for col in cols_to_imput:
        df.loc[df[col].isnull(), col] = df.loc[df[col].isnull(),
                                               'cat_price_median']
    df.fillna(-999, inplace=True)
    df = df[['item_id', 'shop_id', 'date_block_num',
             'item_shop_price_median',
             'cat_price_median',
             'cat_shop_price_median',
             'item_price_median']]

    print("%s columns: %s" % (output_path, str(df.columns)))
    df.to_parquet(output_path)
