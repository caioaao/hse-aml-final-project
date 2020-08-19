from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer


DEFAULT_CORRECTION_CONST = 0.4770088818840332


def corrected_rmse(y, y_pred, c=DEFAULT_CORRECTION_CONST):
    return mean_squared_error(y, y_pred, squared=False) * c


corrected_rmse_score = make_scorer(corrected_rmse, greater_is_better=False)
