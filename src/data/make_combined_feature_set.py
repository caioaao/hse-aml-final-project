import pandas as pd
from functools import reduce
from tqdm.auto import tqdm


def _combine_dfs(df1, df2):
    df = df1.merge(df2, on=['item_id', 'shop_id', 'date_block_num'],
                   suffixes=('', '_to_drop'))
    df.drop(columns=[col for col in df.columns if col.endswith('_to_drop')],
            inplace=True)
    return df


if __name__ == '__main__':
    import sys

    input_dfs_paths = sys.argv[1:-1]
    output_path = sys.argv[-1]
    input_dfs = [pd.read_parquet(p) for p in input_dfs_paths]
    df = reduce(_combine_dfs, tqdm(input_dfs[1:]),
                input_dfs[0])

    print("%s columns: %s" % (output_path, str(df.columns)))
    df.to_parquet(output_path)
