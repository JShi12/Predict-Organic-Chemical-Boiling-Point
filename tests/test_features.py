import pandas as pd

from boiling_point.features import add_smiles_features, drop_uninformative_columns


def test_add_smiles_features_counts_atoms_and_bonds():
    df = pd.DataFrame({
        "isosmiles": ["CC(=O)O", "C#CCN"],
    })

    result = add_smiles_features(df)

    assert result.loc[0, ["C_cnt", "O_cnt", "N_cnt", "F_cnt",
                           "side_chain_cnt", "double_bond_cnt", "triple_bond_cnt"]].tolist() == [
        2, 2, 0, 0, 1, 1, 0,
    ]
    assert result.loc[1, ["C_cnt", "O_cnt", "N_cnt", "F_cnt",
                           "side_chain_cnt", "double_bond_cnt", "triple_bond_cnt"]].tolist() == [
        3, 0, 1, 0, 0, 0, 1,
    ]


def test_drop_uninformative_columns_removes_requested_columns():
    df = pd.DataFrame({"charge": [0, 0], "F_cnt": [0, 1], "mw": [10.0, 20.0]})

    result = drop_uninformative_columns(df)

    assert list(result.columns) == ["mw"]
