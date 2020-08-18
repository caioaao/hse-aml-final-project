import pandas as pd


if __name__ == '__main__':
    import sys
    input_df = pd.read_parquet(sys.argv[1])
    date_ids = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    df = input_df.merge(
        date_ids, on=['date_block_num'],
        how='left', sort=False)

    df.to_parquet(output_path)
