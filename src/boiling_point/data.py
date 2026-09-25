"""Loading and merging the two source datasets."""
import os
import warnings
from pathlib import Path

import pandas as pd

PUBCHEM_COLUMNS = [
    "cmpdname", "mw", "mf", "polararea", "hbonddonor", "hbondacc",
    "rotbonds", "heavycnt", "isosmiles", "charge",
]

# The full PubChem download (~77 MB, not committed) and the committed subset
# of the rows this project uses (built by scripts/build_pubchem_subset.py).
PUBCHEM_FULL_PATH = "compound_property_from_PubChem.csv"
PUBCHEM_SUBSET_PATH = str(Path(__file__).resolve().parents[2] / "data" / "pubchem_subset.csv")


def load_literature_data(path: str) -> pd.DataFrame:
    """Load the literature-sourced boiling point dataset."""
    return pd.read_excel(path)


def load_nist_data(path: str) -> pd.DataFrame:
    """Load boiling points scraped from NIST WebBook (see
    scripts/scrape_nist_boiling_points.py), targeting the high polar-area/
    rotatable-bond region underrepresented in the literature dataset. Same
    schema as load_literature_data (cmpdname, boiling_point_kelvin)."""
    return pd.read_csv(path)


def load_pubchem_data(path: str = PUBCHEM_FULL_PATH, columns=None) -> pd.DataFrame:
    """Load the PubChem physicochemical property dataset, keeping only
    the columns useful for modeling (the raw download also has string
    columns that aren't needed here). If the full download isn't present,
    falls back to the committed subset, which holds every row this project
    uses, so all notebooks and scripts give the same results."""
    if not os.path.exists(path) and os.path.exists(PUBCHEM_SUBSET_PATH):
        warnings.warn(f"{path} not found; using the committed subset {PUBCHEM_SUBSET_PATH}", stacklevel=2)
        path = PUBCHEM_SUBSET_PATH
    df = pd.read_csv(path)
    return df[columns or PUBCHEM_COLUMNS]


def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame, on: str = "cmpdname",
                    how: str = "inner") -> pd.DataFrame:
    """Merge the literature boiling-point data with PubChem properties."""
    return df1.merge(df2, on=on, how=how)


def build_modelling_table(labels_df: pd.DataFrame, pubchem_df: pd.DataFrame) -> pd.DataFrame:
    """Merge boiling-point labels with PubChem properties and add the
    SMILES-derived features -- the same steps the main notebook applies."""
    from .features import add_smiles_features
    merged = merge_datasets(labels_df, pubchem_df).reset_index(drop=True)
    return add_smiles_features(merged)
