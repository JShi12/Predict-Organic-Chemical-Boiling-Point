import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from boiling_point.evaluation import evaluate_tuned_models, repeated_split_comparison, summarize


def _synthetic_data(n=200):
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"a": rng.rand(n), "b": rng.rand(n)})
    y = pd.Series(3 * X["a"] - 2 * X["b"] + rng.normal(scale=0.01, size=n))
    return X, y


def test_repeated_split_comparison_returns_one_row_per_seed_per_model():
    X, y = _synthetic_data()
    estimators = {"Ridge": Ridge()}

    results = repeated_split_comparison(X, y, estimators, seeds=[0, 1, 2])

    assert list(results.columns) == ["seed", "model", "rmse"]
    assert len(results) == 3
    assert (results["rmse"] >= 0).all()


def test_summarize_aggregates_mean_and_std_per_model():
    results = pd.DataFrame({
        "seed": [0, 1, 2],
        "model": ["Ridge", "Ridge", "Ridge"],
        "rmse": [1.0, 2.0, 3.0],
    })

    summary = summarize(results)

    assert summary.loc["Ridge", "mean"] == 2.0
    assert summary.loc["Ridge", "std"] == 1.0


def test_evaluate_tuned_models_returns_one_row_per_model():
    X, y = _synthetic_data()
    ridge = Ridge().fit(X, y)

    # "Random Forest" isn't in SCALED_MODELS, so this exercises the unscaled
    # branch without needing a fitted y-scaler; the dispatch only looks at
    # the dict key, not the estimator's real identity.
    result = evaluate_tuned_models(
        {"Random Forest": ridge}, X_val=X, X_val_scaled=X, y_val=y, scaler_y=None
    )

    assert list(result.index) == ["Random Forest"]
    assert result.loc["Random Forest", "rmse"] >= 0
