"""Split-sensitivity check: how much validation performance varies across
different train/val/test splits, holding each architecture's already-tuned
hyperparameters fixed (found once via GridSearchCV in the main pipeline).
This is much cheaper than re-running a full grid search per split, and it
isolates variance coming from the split itself."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from . import preprocessing

# Model names whose estimators were trained/predict on scaled X and y.
SCALED_MODELS = {"Ridge", "Neural Network", "SVR"}


def repeated_split_comparison(X, y, tuned_estimators: dict, seeds) -> pd.DataFrame:
    """For each seed, split the data, fit a fresh clone of each tuned
    estimator (same hyperparameters, no re-tuning) on that split's training
    set, and record its validation RMSE.

    tuned_estimators: {model_name: fitted best_estimator_ from GridSearchCV}
    """
    rows = []
    for seed in seeds:
        X_train, X_val, _, y_train, y_val, _ = preprocessing.train_val_test_split(
            X, y, random_state=seed
        )
        scaler_X = StandardScaler().fit(X_train)
        X_train_scaled = scaler_X.transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)

        scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))
        y_train_scaled = scaler_y.transform(y_train.values.reshape(-1, 1)).ravel()

        for name, estimator in tuned_estimators.items():
            model = clone(estimator)
            if name in SCALED_MODELS:
                model.fit(X_train_scaled, y_train_scaled)
                pred_scaled = model.predict(X_val_scaled)
                pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
            rows.append({"seed": seed, "model": name, "rmse": rmse})
    return pd.DataFrame(rows)


def evaluate_tuned_models(tuned_estimators: dict, X_val, X_val_scaled, y_val, scaler_y) -> pd.DataFrame:
    """Validation RMSE for a dict of already-fitted best_estimator_ objects
    on one fixed split (handles scaled vs unscaled models via SCALED_MODELS)."""
    rows = []
    for name, model in tuned_estimators.items():
        if name in SCALED_MODELS:
            pred_scaled = model.predict(X_val_scaled)
            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        else:
            pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        rows.append({"model": name, "rmse": rmse})
    return pd.DataFrame(rows).set_index("model")


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean/std validation RMSE per model, sorted best (lowest mean) first."""
    return (
        results.groupby("model")["rmse"]
        .agg(mean="mean", std="std")
        .round(2)
        .sort_values("mean")
    )
