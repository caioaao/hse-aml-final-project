import sys

import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.pipeline import make_pipeline

from ..data import df_to_X_y
from . import tscv

if __name__ == '__main__':
    train_set = pd.read_parquet(sys.argv[1])
    preprocessor = joblib.load(sys.argv[2])
    output_path = sys.argv[3]

    X_train, y_train = df_to_X_y(train_set)
    X_train = np.asfortranarray(X_train)

    cv_splits = tscv.split(train_set['date_block_num'].values, window=16)
    del train_set

    enet_cv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                           fit_intercept=True, normalize=False, eps=1e-5,
                           cv=cv_splits, verbose=2, random_state=83232)

    reg_cv = make_pipeline(preprocessor, enet_cv)

    reg_cv.fit(X_train, y_train)

    joblib.dump(reg_cv, '%s.cv' % output_path)

    enet = ElasticNet(l1_ratio=reg_cv.l1_ratio_,
                      fit_intercept=False, normalize=False, eps=1e-5,
                      verbose=2, random_state=83232)
    reg = make_pipeline(preprocessor, enet)
    print(reg)
    reg.fit(X_train, y_train)

    joblib.dump(reg, output_path)
