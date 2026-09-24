"""Curated RDKit descriptors for boiling-point prediction.

A liquid boils when its molecules overcome the forces holding them
together, so each descriptor is here for the intermolecular force it
tracks (DESCRIPTOR_GROUPS). Computed from the isomeric SMILES; the old
12 features (features.MODEL_FEATURE_COLUMNS) are kept for comparison.

Some descriptors are constant on a given dataset (e.g. halogen counts on
the literature data, which has no halogenated compounds); the modelling
pipeline drops those on the training folds (VarianceThreshold).
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, GraphDescriptors, rdMolDescriptors
from sklearn.base import BaseEstimator, TransformerMixin

_UNSATURATION = {"n_double_CC": Chem.MolFromSmarts("C=C"), "n_triple_CC": Chem.MolFromSmarts("C#C")}
ELEMENTS = ["C", "O", "N", "S", "F", "Cl", "Br", "I", "B"]


def wiener_index(mol) -> float:
    """Sum of shortest-path distances between all pairs of heavy atoms.
    Introduced by Wiener (1947) for alkane boiling points: branched isomers
    are more compact, with a smaller index and a lower boiling point."""
    return float(Chem.GetDistanceMatrix(mol).sum() / 2)


DESCRIPTOR_GROUPS = {
    "size / polarizability (dispersion forces)": {
        "MolWt": Descriptors.MolWt,
        "HeavyAtomCount": Descriptors.HeavyAtomCount,
        "MolMR": Descriptors.MolMR,
        "LabuteASA": rdMolDescriptors.CalcLabuteASA,
    },
    "branching / shape (contact area)": {
        "WienerIndex": wiener_index,
        "Kappa1": rdMolDescriptors.CalcKappa1,
        "Kappa2": rdMolDescriptors.CalcKappa2,
        "Kappa3": rdMolDescriptors.CalcKappa3,
        "Chi0v": rdMolDescriptors.CalcChi0v,
        "Chi1v": rdMolDescriptors.CalcChi1v,
        "BalabanJ": GraphDescriptors.BalabanJ,
    },
    "polarity / hydrogen bonding": {
        "TPSA": rdMolDescriptors.CalcTPSA,
        "MolLogP": Descriptors.MolLogP,
        "NumHDonors": rdMolDescriptors.CalcNumHBD,
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA,
        "fr_Al_OH": Fragments.fr_Al_OH,
        "fr_Ar_OH": Fragments.fr_Ar_OH,
        "fr_COO": Fragments.fr_COO,
        "fr_NH2": Fragments.fr_NH2,
        "fr_NH1": Fragments.fr_NH1,
        "fr_NH0": Fragments.fr_NH0,
        "fr_amide": Fragments.fr_amide,
        "fr_ester": Fragments.fr_ester,
        "fr_ketone": Fragments.fr_ketone,
        "fr_aldehyde": Fragments.fr_aldehyde,
        "fr_ether": Fragments.fr_ether,
        "fr_nitro": Fragments.fr_nitro,
        "fr_nitrile": Fragments.fr_nitrile,
    },
    "rings / flexibility / unsaturation": {
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
        "RingCount": rdMolDescriptors.CalcNumRings,
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
        **{name: (lambda mol, p=pattern: len(mol.GetSubstructMatches(p))) for name, pattern in _UNSATURATION.items()},
    },
    "composition": {
        f"n_{el}": (lambda mol, el=el: sum(atom.GetSymbol() == el for atom in mol.GetAtoms())) for el in ELEMENTS
    },
}
CURATED_COLUMNS = [name for group in DESCRIPTOR_GROUPS.values() for name in group]


def curated_descriptors(smiles) -> pd.DataFrame:
    """One row per SMILES, one column per curated descriptor."""
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES {s!r}")
        rows.append({name: float(fn(mol)) for group in DESCRIPTOR_GROUPS.values() for name, fn in group.items()})
    index = smiles.index if isinstance(smiles, pd.Series) else None
    return pd.DataFrame(rows, columns=CURATED_COLUMNS, index=index)


class DropCorrelated(BaseEstimator, TransformerMixin):
    """Drop features whose absolute correlation with an earlier-kept feature
    exceeds `threshold`. Fitted on training data only (inside the pipeline),
    so it can't peek at validation folds. Near-duplicate features (MolWt,
    MolMR, LabuteASA, Chi0v all track size) destabilise Ridge and the MLP."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        corr = np.atleast_2d(np.abs(np.corrcoef(X, rowvar=False)))
        keep = []
        for j in range(X.shape[1]):
            if all(corr[j, k] <= self.threshold for k in keep):
                keep.append(j)
        self.keep_ = np.array(keep)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]
