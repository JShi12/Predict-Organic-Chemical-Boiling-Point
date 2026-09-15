import pandas as pd

from boiling_point.features import get_target, select_model_features
from boiling_point.preprocessing import train_val_test_split


def _synthetic_df(n=100):
    return pd.DataFrame({
        "mw": range(n),
        "polararea": range(n),
        "hbonddonor": range(n),
        "hbondacc": range(n),
        "rotbonds": range(n),
        "heavycnt": range(n),
        "C_cnt": range(n),
        "O_cnt": range(n),
        "N_cnt": range(n),
        "side_chain_cnt": range(n),
        "double_bond_cnt": range(n),
        "triple_bond_cnt": range(n),
        "boiling_point_kelvin": [float(i) for i in range(n)],
    })


def test_train_val_test_split_proportions_and_no_overlap():
    df = _synthetic_df(100)
    X = select_model_features(df)
    y = get_target(df)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    assert len(X_test) == 20
    assert len(X_val) == 20
    assert len(X_train) == 60
    assert len(y_train) == 60 and len(y_val) == 20 and len(y_test) == 20

    train_idx, val_idx, test_idx = set(X_train.index), set(X_val.index), set(X_test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_select_model_features_excludes_target():
    df = _synthetic_df(10)

    X = select_model_features(df)
    y = get_target(df)

    assert "boiling_point_kelvin" not in X.columns
    assert len(y) == 10
