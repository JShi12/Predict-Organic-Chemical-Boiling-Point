"""Loading and merging the two source datasets."""
import pandas as pd

PUBCHEM_COLUMNS = [
    "cmpdname", "mw", "mf", "polararea", "hbonddonor", "hbondacc",
    "rotbonds", "heavycnt", "isosmiles", "charge",
]


def load_literature_data(path: str) -> pd.DataFrame:
    """Load the literature-sourced boiling point dataset."""
    return pd.read_excel(path)


def load_pubchem_data(path: str, columns=None) -> pd.DataFrame:
    """Load the PubChem physicochemical property dataset, keeping only
    the columns useful for modeling (the raw download also has string
    columns that aren't needed here)."""
    df = pd.read_csv(path)
    return df[columns or PUBCHEM_COLUMNS]


def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame, on: str = "cmpdname",
                    how: str = "inner") -> pd.DataFrame:
    """Merge the literature boiling-point data with PubChem properties."""
    return df1.merge(df2, on=on, how=how)
