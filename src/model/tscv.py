import pandas as pd


def split(df: pd.DataFrame):
    return [[df.index[df['date_block_num'] < k],
             df.index[df['date_block_num'] == k]]
            for k in [31, 32, 33]]
