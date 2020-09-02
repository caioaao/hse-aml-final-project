import pandas as pd

if __name__ == '__main__':
    import sys

    input_dfs_paths = sys.argv[1:-1]
    output_path = sys.argv[-1]

    input_dfs = [pd.read_parquet(p) for p in input_dfs_paths]

    input_dfs = [input_dfs[0]] + [input_df.drop(columns=input_dfs[0].columns)
                                  for input_df in input_dfs[1:]]

    pd.concat(input_dfs, axis=1).to_parquet(output_path)
