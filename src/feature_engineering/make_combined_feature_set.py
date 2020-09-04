import pandas as pd

if __name__ == '__main__':
    import sys

    input_dfs_paths = sys.argv[1:-1]
    output_path = sys.argv[-1]

    input_dfs = [pd.read_parquet(p) for p in input_dfs_paths]
    prev_cols = []
    for df in input_dfs:
        df.drop(columns=prev_cols, errors='ignore', inplace=True)
        prev_cols = prev_cols + list(df.columns)

    pd.concat(input_dfs, axis=1).to_parquet(output_path)
