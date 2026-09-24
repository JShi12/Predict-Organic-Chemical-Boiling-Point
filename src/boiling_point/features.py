"""SMILES-derived feature engineering and feature/target selection."""
import pandas as pd

MODEL_FEATURE_COLUMNS = [
    "mw", "polararea", "hbonddonor", "hbondacc", "rotbonds", "heavycnt",
    "C_cnt", "O_cnt", "N_cnt", "side_chain_cnt", "double_bond_cnt",
    "triple_bond_cnt",
]


def add_smiles_features(df: pd.DataFrame, smiles_col: str = "isosmiles") -> pd.DataFrame:
    """Derive simple atom/bond-count features from the isomeric SMILES
    string (background-knowledge features: these characters map 1:1 to
    atom/bond counts in SMILES notation)."""
    df = df.copy()
    df["C_cnt"] = df[smiles_col].apply(lambda x: x.count("C"))
    df["O_cnt"] = df[smiles_col].apply(lambda x: x.count("O"))
    df["N_cnt"] = df[smiles_col].apply(lambda x: x.count("N"))
    df["F_cnt"] = df[smiles_col].apply(lambda x: x.count("F"))
    df["side_chain_cnt"] = df[smiles_col].apply(lambda x: x.count("("))
    df["double_bond_cnt"] = df[smiles_col].apply(lambda x: x.count("="))
    df["triple_bond_cnt"] = df[smiles_col].apply(lambda x: x.count("#"))
    return df


def drop_uninformative_columns(df: pd.DataFrame, cols=("charge", "F_cnt")) -> pd.DataFrame:
    """Drop columns found to have no correlation with the target during EDA."""
    return df.drop(list(cols), axis=1)


def select_model_features(df: pd.DataFrame, feature_cols=None) -> pd.DataFrame:
    """Select the numeric predictor columns used for model training."""
    return df[feature_cols or MODEL_FEATURE_COLUMNS].copy()


def in_hard_region(df: pd.DataFrame, polararea_threshold: float = 40,
                    rotbonds_threshold: float = 14) -> pd.Series:
    """Boolean mask for the underrepresented outlier region from the error
    analysis: high polar area and/or many rotatable bonds."""
    return (df["polararea"] >= polararea_threshold) | (df["rotbonds"] >= rotbonds_threshold)


def get_target(df: pd.DataFrame, target_col: str = "boiling_point_kelvin") -> pd.Series:
    return df[target_col].copy()
