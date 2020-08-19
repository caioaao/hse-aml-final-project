import pandas as pd
from zipfile import ZipFile


def _load_raw_datasets(raw_data_path):
    with ZipFile(raw_data_path, 'r') as datasets_file:
        return (pd.read_csv(datasets_file.open('sales_train.csv')),
                pd.read_csv(datasets_file.open('test.csv')))


if __name__ == '__main__':
    import sys
    raw_data_path = sys.argv[1]
    output_path = sys.argv[2]
    sales_train, test_set = _load_raw_datasets(raw_data_path)
    ids_from_train = sales_train.loc[
        sales_train['date_block_num'] == 33, ['item_id', 'shop_id']]\
        .drop_duplicates()
    test_set = test_set.merge(ids_from_train, on=['item_id', 'shop_id'])
    test_set['date_block_num'] = sales_train['date_block_num'].max() + 1
    test_set[['item_id', 'shop_id', 'date_block_num']].to_parquet(output_path)
