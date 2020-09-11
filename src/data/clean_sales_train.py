import sys
import pandas as pd
import zipfile

if __name__ == '__main__':
    raw_data_path = sys.argv[1]
    output_path = sys.argv[2]

    with zipfile.ZipFile(raw_data_path, 'r') as datasets_file:
        sales_train = pd.read_csv(datasets_file.open('sales_train.csv'))

    sales_train = sales_train[(sales_train['item_price'] > 0)
                              & (sales_train['item_price'] < 3e5)]

    sales_train.to_parquet(output_path)
