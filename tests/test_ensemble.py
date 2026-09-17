import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from boiling_point.ensemble import AverageEnsembleRegressor


def _synthetic_data(n=100):
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"a": rng.rand(n), "b": rng.rand(n)})
    y = pd.Series(3 * X["a"] - 2 * X["b"] + rng.normal(scale=0.01, size=n))
    return X, y


def test_fit_predict_returns_one_prediction_per_row():
    X, y = _synthetic_data()
    ensemble = AverageEnsembleRegressor(
        ridge=Ridge(),
        xgb=XGBRegressor(objective="reg:squarederror", random_state=0, n_estimators=10),
        nn=MLPRegressor(max_iter=200, random_state=0),
    )

    ensemble.fit(X, y)
    predictions = ensemble.predict(X)

    assert len(predictions) == len(X)


def test_predict_is_the_mean_of_the_three_members():
    X, y = _synthetic_data()
    ridge = Ridge()
    xgb = XGBRegressor(objective="reg:squarederror", random_state=0, n_estimators=10)
    nn = MLPRegressor(max_iter=200, random_state=0)
    ensemble = AverageEnsembleRegressor(ridge=ridge, xgb=xgb, nn=nn)
    ensemble.fit(X, y)

    ensemble_pred = ensemble.predict(X)

    ridge_pred = ensemble.scaler_y_.inverse_transform(
        ensemble.ridge_.predict(ensemble.scaler_X_.transform(X)).reshape(-1, 1)
    ).ravel()
    xgb_pred = ensemble.xgb_.predict(X)
    nn_pred = ensemble.scaler_y_.inverse_transform(
        ensemble.nn_.predict(ensemble.scaler_X_.transform(X)).reshape(-1, 1)
    ).ravel()
    expected = (ridge_pred + xgb_pred + nn_pred) / 3

    np.testing.assert_allclose(ensemble_pred, expected)


def test_is_clonable_for_use_with_sklearn_utilities():
    from sklearn.base import clone

    ensemble = AverageEnsembleRegressor(
        ridge=Ridge(),
        xgb=XGBRegressor(objective="reg:squarederror", random_state=0, n_estimators=10),
        nn=MLPRegressor(max_iter=200, random_state=0),
    )

    cloned = clone(ensemble)

    assert isinstance(cloned, AverageEnsembleRegressor)
    assert cloned is not ensemble
