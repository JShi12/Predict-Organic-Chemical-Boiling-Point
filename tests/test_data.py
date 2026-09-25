import pandas as pd
import pytest

from boiling_point import data


def test_load_pubchem_data_falls_back_to_the_committed_subset(tmp_path, monkeypatch):
    subset = tmp_path / "subset.csv"
    pd.DataFrame({c: [1] for c in ["cid"] + data.PUBCHEM_COLUMNS}).to_csv(subset, index=False)
    monkeypatch.setattr(data, "PUBCHEM_SUBSET_PATH", str(subset))

    with pytest.warns(UserWarning, match="committed subset"):
        df = data.load_pubchem_data(str(tmp_path / "missing_full_download.csv"))

    assert list(df.columns) == data.PUBCHEM_COLUMNS


def test_committed_subset_exists_with_the_modelling_columns():
    df = pd.read_csv(data.PUBCHEM_SUBSET_PATH, nrows=5)

    assert set(data.PUBCHEM_COLUMNS) <= set(df.columns)
