import sys
import pandas as pd

MONTHS_TO_DROP = [5, 7, 8, 9, 11, 17, 19, 21, 22, 23, 25, 27, 29]

if __name__ == '__main__':
    df = pd.read_parquet(sys.argv[1])
    output_path = sys.argv[2]

    df.drop(columns=['f__item_cnt_%d' % i for i in MONTHS_TO_DROP])\
      .to_parquet(output_path)
