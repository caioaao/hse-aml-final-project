import logging

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    import sys

    cv_df = pd.read_parquet(sys.argv[1])
    output_path = sys.argv[2]

    logging.basicConfig(handlers=[logging.FileHandler(output_path),
                                  logging.StreamHandler()],
                        force=True, level=logging.INFO,
                        format='%(asctime)-15s %(message)s')

    scores = []
    for fold_id in cv_df['fold_id'].unique():
        y_true = cv_df[cv_df['fold_id'] == fold_id]['item_cnt_month'].values
        y_pred = cv_df[cv_df['fold_id'] == fold_id]['oof_preds'].values
        err = mean_squared_error(y_true, np.clip(y_pred, 0, 20), squared=False)
        logging.info('Fold %d: RMSE: %.5f' % (fold_id, err))
        scores.append(err)
    logging.info('CV SCORE: %.5f (std %.5f)' % (np.mean(scores),
                                                np.std(scores)))
