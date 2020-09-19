import sys

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from ..data import add_as_features


def _combine_cv_preds(df1: pd.DataFrame, df2: pd.DataFrame):
    df = df1.merge(df2, on=['oof_idx', 'fold_id', 'item_cnt_month', 'date_block_num'])
    if not (df.shape[0] == df1.shape[0]
            and df.shape[0] == df2.shape[0]):
        raise ValueError("CV preds don't align")
    return df


def _prepare_df(df, idx):
    max_idx = df['fold_id'].max()
    df['date_block_num'] = 33 - max_idx + df['fold_id']
    df['oof_preds'] = np.clip(df['oof_preds'], 0, 20)
    df.rename(columns={'oof_preds': 'oof_preds_%d' % idx}, inplace=True)

    return df


if __name__ == '__main__':
    output_path = sys.argv[-1]
    input_df_paths = sys.argv[1:-1]

    df = _prepare_df(pd.read_parquet(input_df_paths[0]), 0)
    for i, path in tqdm(enumerate(input_df_paths[1:])):
        df = _combine_cv_preds(df, _prepare_df(pd.read_parquet(path), i + 1))

    add_as_features(df, [col for col in df.columns
                         if col.startswith('oof_preds')],
                    inplace=True)

    df.to_parquet(output_path)
