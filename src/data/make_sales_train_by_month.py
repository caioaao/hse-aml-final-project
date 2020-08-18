from ..functional import comp, partial

import pandas as pd
import numpy as np
from zipfile import ZipFile


def _load_raw(raw_data_path):
    with ZipFile(raw_data_path, 'r') as datasets_file:
        return pd.read_csv(datasets_file.open('sales_train.csv'))


def _filter_dropped_samples(df):
    return df[~((df['shop_id'] < 2)
                | (df['item_id'] < 30)
                | (df['item_id'] > 22167))]


def _group_by_month(df):
    df2 = df.groupby(
        by=['date_block_num', 'shop_id', 'item_id'])['item_cnt_day']\
            .sum().reset_index()
    df2.rename(columns={'item_cnt_day': 'item_cnt'}, inplace=True)
    return df2


def _clip_count(df):
    df = df.copy()
    df['item_cnt'] = np.clip(df['item_cnt'], 0, 20)
    return df


def _save_result(output_path, df):
    df.to_parquet(output_path)


if __name__ == '__main__':
    import sys
    raw_data_path = sys.argv[1]
    output_path = sys.argv[2]
    comp(partial(_save_result, output_path),
         _clip_count,
         _group_by_month,
         _filter_dropped_samples)(_load_raw(raw_data_path))
