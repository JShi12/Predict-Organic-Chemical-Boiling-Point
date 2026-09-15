"""Model training (GridSearchCV per architecture) and evaluation helpers."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

REGRESSION_SCORING = {
    "neg_mean_squared_error": "neg_mean_squared_error",
    "neg_mean_absolute_error": "neg_mean_absolute_error",
    "r2": "r2",
}

RIDGE_PARAM_GRID = {"alpha": [0.1, 1, 10, 100, 1000]}

RF_PARAM_GRID = {
    "max_depth": [6, 10, None],
    "max_features": ["sqrt", 0.6],
    "max_samples": [0.9],
    "min_samples_leaf": [1, 2],
    "min_samples_split": [2, 3],
    "n_estimators": [75, 100, 200],
}

XGB_PARAM_GRID = {
    "max_depth": [3, 5, 7],
    "min_child_weight": [3, 5],
    "learning_rate": [0.01, 0.1],
    "n_estimators": [300, 500],
}

# Slightly wider max_depth range used when retraining the champion model
# on train+val data (matches the notebook's retrain step).
XGB_RETRAIN_PARAM_GRID = {
    "max_depth": [3, 6, 8],
    "min_child_weight": [3, 5],
    "learning_rate": [0.01, 0.1],
    "n_estimators": [300, 500],
}

MLP_PARAM_GRID = {
    "hidden_layer_sizes": [(50,), (100, 50), (100, 50, 20)],
    "activation": ["relu", "tanh"],
    "solver": ["adam", "sgd"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate": ["constant", "adaptive"],
    "learning_rate_init": [0.001, 0.01, 0.1],
}

SVR_PARAM_GRID = {
    "kernel": ["rbf"],
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.1, 1],
    "epsilon": [0.1, 0.2, 0.5],
}


def train_ridge_cv(X_train_scaled, y_train_scaled, param_grid=None, cv=5):
    grid = GridSearchCV(Ridge(), param_grid or RIDGE_PARAM_GRID, cv=cv,
                         scoring="neg_mean_squared_error", verbose=1)
    grid.fit(X_train_scaled, y_train_scaled)
    return grid


def train_random_forest_cv(X_train, y_train, param_grid=None, cv=5):
    grid = GridSearchCV(
        RandomForestRegressor(random_state=0), param_grid or RF_PARAM_GRID,
        scoring=REGRESSION_SCORING, cv=cv, refit="neg_mean_squared_error", verbose=1,
    )
    grid.fit(X_train, y_train)
    return grid


def train_xgboost_cv(X_train, y_train, param_grid=None, cv=5):
    grid = GridSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=0),
        param_grid or XGB_PARAM_GRID,
        scoring=REGRESSION_SCORING, cv=cv, refit="neg_mean_squared_error", verbose=1,
    )
    grid.fit(X_train, y_train)
    return grid


def train_mlp_cv(X_train_scaled, y_train_scaled, param_grid=None, cv=5):
    grid = GridSearchCV(
        MLPRegressor(max_iter=1000, tol=1e-4), param_grid or MLP_PARAM_GRID,
        cv=cv, scoring="neg_mean_squared_error", verbose=1, n_jobs=-1,
    )
    grid.fit(X_train_scaled, y_train_scaled)
    return grid


def train_svr_cv(X_train_scaled, y_train_scaled, param_grid=None, cv=5):
    grid = GridSearchCV(
        SVR(), param_grid or SVR_PARAM_GRID, cv=cv,
        scoring="neg_mean_squared_error", verbose=1, n_jobs=-1,
    )
    grid.fit(X_train_scaled, y_train_scaled)
    return grid


def rmse_from_cv_unscaled(grid_search: GridSearchCV) -> float:
    """RMSE from a GridSearchCV fit directly on unscaled y (tree models)."""
    return float(np.sqrt(-grid_search.best_score_))


def rmse_from_cv_scaled(grid_search: GridSearchCV, y_train) -> float:
    """RMSE from a GridSearchCV fit on scaled y: undoes the StandardScaler
    by rescaling the MSE with the original target's variance."""
    mse_scaled = -grid_search.best_score_
    mse_unscaled = mse_scaled * np.var(y_train)
    return float(np.sqrt(mse_unscaled))


def retrain_champion(X_train_val, y_train_val, param_grid=None, cv=5):
    """Retrain the champion (XGBoost) architecture on train+val combined."""
    grid = GridSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=0),
        param_grid or XGB_RETRAIN_PARAM_GRID,
        scoring=REGRESSION_SCORING, cv=cv, refit="neg_mean_squared_error", verbose=1,
    )
    grid.fit(X_train_val, y_train_val)
    return grid


def evaluate_on_test(model, X_test, y_test) -> dict:
    """Return {rmse, mae, r2} for a fitted model on the held-out test set."""
    y_pred = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
