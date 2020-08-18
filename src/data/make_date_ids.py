import pandas as pd
import numpy as np

if __name__ == '__main__':
    import sys
    train_set = pd.read_parquet(sys.argv[1])
    output_path = sys.argv[2]
    date_block_nums = np.arange(0, train_set['date_block_num'].max() + 2)
    date_ids = pd.DataFrame({'date_block_num': date_block_nums,
                             'month_id': date_block_nums % 12,
                             'year_id': date_block_nums // 12})
    date_ids.to_parquet(output_path)
