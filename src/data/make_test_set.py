import pandas as pd
from zipfile import ZipFile


if __name__ == '__main__':
    import sys
    raw_data_path = sys.argv[1]
    output_path = sys.argv[2]
    with ZipFile(raw_data_path, 'r') as datasets_file:
        test_set = pd.read_csv(datasets_file.open('test.csv'))
    test_set['date_block_num'] = 34
    test_set.to_parquet(output_path)
