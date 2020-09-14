import joblib
import sys
import pandas as pd

from .linear_preprocess import make_preprocessor

if __name__ == '__main__':
    train_set = pd.read_parquet(sys.argv[1])
    test_set = pd.read_parquet(sys.argv[2])
    output_path = sys.argv[3]

    preprocessor = make_preprocessor(train_set, test_set)

    print(preprocessor)
    joblib.dump(preprocessor, output_path)
