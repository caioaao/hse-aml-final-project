import joblib
import pandas as pd
import numpy as np

from ..feature_engineering import drop_non_features


if __name__ == '__main__':
    import sys
    model = joblib.load(sys.argv[1])
    test_set = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    X_test = drop_non_features(test_set).values
    test_set['item_cnt_month'] = np.clip(model.predict(X_test), 0, 20)
    test_set[['ID', 'item_cnt_month']].to_csv(output_path, index=False)