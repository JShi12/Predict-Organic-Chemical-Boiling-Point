"""Simple-averaging ensemble across heterogeneous model families.

The split-sensitivity check shows no single architecture is reliably best on
this dataset -- validation RMSE ranges overlap heavily across all five. Since
picking a single "champion" partly reflects split luck, this averages
predictions from one representative of each genuinely different inductive
bias: Ridge (linear/regularized), XGBoost (tree/boosting), and a Neural
Network (nonlinear). Random Forest is left out as redundant with XGBoost
(same family), and SVR for being both the least stable across splits and the
only architecture whose train/validation gap didn't improve with more
training data.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import StandardScaler


class AverageEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Averages predictions from a Ridge, an XGBoost, and an MLP regressor.

    Ridge and the MLP are trained on standardized X/y (as elsewhere in this
    project); XGBoost is trained on raw X/y. Scaling is handled internally
    so this behaves as a single sklearn-compatible estimator (fit/predict,
    clonable) usable anywhere a single tuned model was used before.
    """

    def __init__(self, ridge, xgb, nn):
        self.ridge = ridge
        self.xgb = xgb
        self.nn = nn

    def fit(self, X, y):
        y_arr = np.asarray(y).reshape(-1, 1)
        self.scaler_X_ = StandardScaler().fit(X)
        self.scaler_y_ = StandardScaler().fit(y_arr)

        X_scaled = self.scaler_X_.transform(X)
        y_scaled = self.scaler_y_.transform(y_arr).ravel()

        self.ridge_ = clone(self.ridge).fit(X_scaled, y_scaled)
        self.xgb_ = clone(self.xgb).fit(X, y)
        self.nn_ = clone(self.nn).fit(X_scaled, y_scaled)
        return self

    def predict(self, X):
        X_scaled = self.scaler_X_.transform(X)
        ridge_pred = self.scaler_y_.inverse_transform(
            self.ridge_.predict(X_scaled).reshape(-1, 1)
        ).ravel()
        nn_pred = self.scaler_y_.inverse_transform(
            self.nn_.predict(X_scaled).reshape(-1, 1)
        ).ravel()
        xgb_pred = self.xgb_.predict(X)
        return (ridge_pred + xgb_pred + nn_pred) / 3
