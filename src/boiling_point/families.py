"""Assign each compound one primary chemical family from its structure.

Families are checked in priority order, roughly by the strength of the
intermolecular force the group adds -- so a multifunctional compound (an
amino acid, a hydroxy ester) goes to the group that dominates its boiling
point. Used to stratify cross-validation folds and to report errors per
family; the global model doesn't see the family unless it is added as an
explicit input.
"""
import pandas as pd
from rdkit import Chem

# (family, SMARTS patterns -- any match assigns the family), highest priority first
FAMILY_PATTERNS = [
    ("carboxylic acid", ["[CX3](=O)[OX2H1]"]),
    ("amide", ["[NX3][CX3]=[OX1]"]),
    ("alcohol/phenol", ["[OX2H][#6;!$(C=O)]"]),
    ("amine", ["[NX3;!$(N-C=[O,S]);!$(N-[#7,#8,#16])]"]),
    ("ester", ["[#6][CX3](=O)[OX2][#6]"]),
    ("ketone/aldehyde", ["[#6][CX3](=O)[#6]", "[CX3H1](=O)"]),
    ("ether", ["[OD2]([#6])[#6]"]),
    ("nitrile/nitro", ["[CX2]#[NX1]", "[N+](=O)[O-]", "[NX3](=O)=O"]),
    ("sulfur compound", ["[#16]"]),
    ("halide", ["[F,Cl,Br,I]"]),
    ("aromatic", ["a"]),
    ("alkene/alkyne", ["[CX3]=[CX3]", "[CX2]#[CX2]"]),
]
_COMPILED = [(family, [Chem.MolFromSmarts(p) for p in patterns]) for family, patterns in FAMILY_PATTERNS]


def assign_family(smiles: str) -> str:
    """Primary family of one compound; 'alkane' if it contains only carbon
    and hydrogen and nothing above matched, otherwise 'other'."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    for family, patterns in _COMPILED:
        if any(mol.HasSubstructMatch(p) for p in patterns):
            return family
    if all(atom.GetSymbol() == "C" for atom in mol.GetAtoms()):
        return "alkane"
    return "other"


def assign_families(smiles) -> pd.Series:
    smiles = pd.Series(smiles)
    return smiles.map(assign_family).rename("family")


def merge_rare_families(families: pd.Series, min_count: int = 25) -> pd.Series:
    """Fold families with fewer than min_count members into 'other', so every
    family is large enough to appear in every stratified fold."""
    counts = families.value_counts()
    rare = counts[counts < min_count].index
    return families.where(~families.isin(rare), "other")
