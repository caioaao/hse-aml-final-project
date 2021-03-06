import pandas as pd

from . import add_lagged_features, add_as_features


if __name__ == '__main__':
    import sys
    df = pd.read_parquet(sys.argv[1])
    history_df = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    indicators = ['MOEX', 'CNYRUB', 'USDRUB', 'EURRUB']

    for indicator in indicators:
        history_df['%s_gain' % indicator] = (
            history_df['%s_close' % indicator]
            / history_df['%s_open' % indicator])

    feature_cols = ['%s_gain' % indicator for indicator in indicators]
    df = add_lagged_features(df, history_df, feature_cols, max_lag=5,
                             index_cols=[])
    df = df.merge(history_df, on='date_block_num', how='left', sort=False)

    add_as_features(df, feature_cols, inplace=True)
    print("%s columns: %s" % (output_path, str(df.columns)))
    df.to_parquet(output_path)
