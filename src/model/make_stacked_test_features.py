import sys
import pandas as pd
from tqdm.auto import tqdm

from ..data import add_as_features


def prepare_df(df, idx):
    df.rename(columns={'item_cnt_month': 'pred_%d' % idx}, inplace=True)
    return df[['ID', 'pred_%d' % idx]]


if __name__ == '__main__':
    input_paths = sys.argv[1:-1]
    output_path = sys.argv[-1]

    df = prepare_df(pd.read_parquet(input_paths[0]), 0)
    for i, path in tqdm(enumerate(input_paths[1:])):
        df = df.merge(prepare_df(pd.read_parquet(path), i+1), on='ID')

    add_as_features(df, [col for col in df.columns if col.startswith('pred_')],
                    inplace=True)

    print(df.columns)

    df.to_parquet(output_path)
