"""Train/val/test splitting and feature/target scaling."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_val_test_split(X, y, test_size: float = 0.2, val_size: float = 0.25,
                          random_state: int = 42):
    """Two-stage split giving a 60/20/20 train/val/test ratio by default
    (test_size carves off the test set, then val_size carves the
    validation set out of what remains)."""
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=val_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Fit a StandardScaler on X_train and apply it to all three splits."""
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler_X


def scale_target(y_train, y_val, y_test):
    """Fit a StandardScaler on y_train and apply it to all three splits."""
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
    return y_train_scaled, y_val_scaled, y_test_scaled, scaler_y
