import pandas as pd
from zipfile import ZipFile


if __name__ == '__main__':
    import sys
    raw_data_path = sys.argv[1]
    output_path = sys.argv[2]
    with ZipFile(raw_data_path, 'r') as datasets_file:
        pd.read_csv(datasets_file.open('test.csv')).to_parquet(output_path)
