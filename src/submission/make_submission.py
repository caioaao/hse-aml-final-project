import joblib
import pandas as pd
import numpy as np

from ..feature_engineering import drop_non_features
from . import submission_from_subset

if __name__ == '__main__':
    import sys
    model = joblib.load(sys.argv[1])
    test_set = pd.read_parquet(sys.argv[2])
    test_subset = pd.read_parquet(sys.argv[3])
    output_path = sys.argv[4]

    X_test = drop_non_features(test_subset).values
    test_subset['item_cnt_month'] = np.clip(model.predict(X_test), 0, 20)
    submission_from_subset(test_subset, test_set).to_csv(output_path,
                                                         index=False)
