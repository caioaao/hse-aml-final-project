import numpy as np
from sklearn.metrics import mean_squared_error


def _clipped_rmse(y, ypred):
    return mean_squared_error(y, np.clip(ypred, 0, 20), squared=False)


def early_stopping_fit(
        reg, X_train, y_train, X_val, y_val, early_stop_rounds=10,
        tol=0.0001, callback=None, max_iter=200):
    best_loss = np.inf
    last_visible_improvement = 1
    for iter_n in range(max_iter):
        reg.partial_fit(X_train, y_train)
        intermediate_value = _clipped_rmse(y_val, reg.predict(X_val))
        print(' iter %2d | val-rmse %8.5f' % (iter_n, intermediate_value))
        if callback:
            callback(reg, iter_n, intermediate_value)
        if best_loss > intermediate_value + tol:
            last_visible_improvement = iter_n + 1
            best_loss = intermediate_value

        if iter_n - last_visible_improvement > early_stop_rounds:
            break
    return reg, last_visible_improvement
