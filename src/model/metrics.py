from sklearn.metrics import mean_squared_error


DEFAULT_CORRECTION_CONST = 0.4770088818840332


def corrected_rmse(y_true, y_pred, c=DEFAULT_CORRECTION_CONST):
    return mean_squared_error(y_true, y_pred, squared=False) * c


def neg_corrected_rmse(y_true, y_pred, c=DEFAULT_CORRECTION_CONST):
    return -corrected_rmse(y_true, y_pred, c=c)
