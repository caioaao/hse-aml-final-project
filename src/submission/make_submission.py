import sys
import pandas as pd


if __name__ == '__main__':
    submission_df = pd.read_parquet(sys.argv[1])
    output_path = sys.argv[2]

    submission_df[['ID', 'item_cnt_month']].to_csv(output_path, index=False)
