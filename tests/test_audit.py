import numpy as np
import pandas as pd
import pytest

from boiling_point.audit import consistent_disagreement, joback_matches, joback_tb


@pytest.mark.parametrize("smiles, expected", [
    ("CCCC", 198.2 + 2 * 23.58 + 2 * 22.88),          # n-butane: 2 CH3 + 2 CH2
    ("CCO", 198.2 + 23.58 + 22.88 + 92.88),           # ethanol: CH3 + CH2 + OH
    ("CC(C)(C)C", 198.2 + 4 * 23.58 + 18.25),         # neopentane: quaternary C
    ("C=CC#C", 198.2 + 18.18 + 24.96 + 27.38 + 9.20),  # vinylacetylene
    ("CCN(C)C", 198.2 + 3 * 23.58 + 22.88 + 11.74),   # tertiary amine
])
def test_joback_tb_sums_group_contributions(smiles, expected):
    assert joback_tb(smiles) == pytest.approx(expected)


@pytest.mark.parametrize("smiles", ["C1=CC=CC=C1", "CCCl", "CC(C)=O", "[2H]C([2H])([2H])[2H]", "B(C)(C)C"])
def test_joback_tb_is_none_for_groups_not_implemented(smiles):
    assert joback_tb(smiles) is None


def test_joback_matches_flags_only_labels_identical_to_the_estimate():
    df = pd.DataFrame({"cmpdname": ["a", "b", "c"], "isosmiles": ["CCCC", "CCCC", "CCO"],
                       "boiling_point_kelvin": [291.12, 272.65, 337.55]})

    flags = joback_matches(df)["label_is_joback"].tolist()

    assert flags == [True, False, False]   # 0.01 K off is treated as a real measurement


def test_consistent_disagreement_needs_every_prediction_off_in_the_same_direction():
    preds = pd.DataFrame({
        "row": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        "y_true": [300.0] * 9,
        "y_pred": [400, 380, 360,      # always ~60-100 K too high -> flagged
                   400, 250, 380,      # misses in both directions -> not flagged
                   320, 330, 310],     # small misses -> not flagged
    })

    flagged = consistent_disagreement(preds, min_error=50)

    assert flagged.index.tolist() == [0]
    assert flagged.loc[0, "mean_residual"] == pytest.approx(80)


def test_implausible_hydrocarbons_uses_the_n_alkane_with_the_same_carbon_count():
    from boiling_point.audit import implausible_hydrocarbons
    df = pd.DataFrame({"cmpdname": ["decane-like", "too-low decyne", "decanol"],
                       "isosmiles": ["CCCCCCCCCC", "CCC#CCCCCCC", "CCCCCCCCCCO"],
                       "boiling_point_kelvin": [447.0, 320.3, 300.0]})

    flagged = implausible_hydrocarbons(df)

    assert flagged["cmpdname"].tolist() == ["too-low decyne"]   # decanol isn't a hydrocarbon
    assert flagged["below_n_alkane"].iloc[0] == pytest.approx(447.3 - 320.3)


def test_build_label_audit_merges_checks_and_marks_exclusions():
    from boiling_point.audit import build_label_audit
    df = pd.DataFrame({"cmpdname": ["butane (Joback)", "bad decyne", "odd amine", "fine alcohol", "typo"],
                       "isosmiles": ["CCCC", "CCC#CCCCCCC", "CCN(CC)CC", "CCCO", "CCCCO"],
                       "boiling_point_kelvin": [291.12, 320.3, 300.0, 370.0, 250.0]})
    nist = pd.DataFrame({"cmpdname": ["bad decyne", "odd amine", "fine alcohol"],
                         "nist_tb": [449.0, None, 370.3]})

    audit = build_label_audit(df, nist, {"typo": "source says 390 K"}).set_index("cmpdname")

    assert audit.loc["butane (Joback)", "category"] == "Joback estimate"
    assert audit.loc["bad decyne", "category"] == "implausible hydrocarbon; contradicted by NIST"
    assert audit.loc["odd amine", "category"] == "model-flagged, unverified"
    assert audit.loc["fine alcohol", "category"] == "NIST agrees"
    assert audit["exclude"].to_dict() == {"butane (Joback)": True, "bad decyne": True, "typo": True,
                                          "odd amine": False, "fine alcohol": False}


@pytest.mark.parametrize("smiles, mirror", [
    ("C[C@@H](O)[C@@H](C)O", "C[C@H](O)[C@H](C)O"),   # (R,R) -> (S,S)
    ("C[C@H](O)[C@@H](C)O", "C[C@H](O)[C@@H](C)O"),   # meso is its own mirror image
    ("OC/C=C/CO", "OC/C=C/CO"),                        # double-bond geometry is unchanged
])
def test_mirror_image_smiles_inverts_only_tetrahedral_centres(smiles, mirror):
    from boiling_point.audit import mirror_image_smiles
    assert mirror_image_smiles(smiles) == mirror


def test_stereoisomer_conflicts_excludes_a_label_contradicting_its_enantiomer():
    from boiling_point.audit import stereoisomer_conflicts
    df = pd.DataFrame({"cmpdname": ["unspecified", "meso", "R,R", "S,S"],
                       "isosmiles": ["CC(O)C(C)O", "C[C@H](O)[C@@H](C)O", "C[C@@H](O)[C@@H](C)O",
                                     "C[C@H](O)[C@H](C)O"],
                       "boiling_point_kelvin": [455.15, 456.65, 350.5, 453.65]})

    out = stereoisomer_conflicts(df).set_index("cmpdname")

    assert out.index.tolist() == ["R,R"]                 # the consistent labels aren't flagged
    assert out.loc["R,R", "category"] == "stereoisomer conflict"
    assert out.loc["R,R", "exclude"]
    assert "enantiomer S,S is 453.65 K" in out.loc["R,R", "evidence"]


def test_stereoisomer_conflicts_only_flags_cis_trans_and_two_label_conflicts():
    from boiling_point.audit import stereoisomer_conflicts
    df = pd.DataFrame({"cmpdname": ["trans", "cis", "R", "S", "unrelated"],
                       "isosmiles": ["OC/C=C/CO", "OC/C=C\\CO", "C[C@@H](O)CC", "C[C@H](O)CC", "CCCO"],
                       "boiling_point_kelvin": [404.15, 508.15, 372.0, 300.0, 370.0]})

    out = stereoisomer_conflicts(df).set_index("cmpdname")

    # cis/trans can genuinely differ; an enantiomer pair alone gives no majority
    assert sorted(out.index) == ["R", "S", "cis", "trans"]
    assert (out["category"] == "stereoisomer conflict, unverified").all()
    assert not out["exclude"].any()


def test_stereoisomer_conflicts_ignores_spreads_within_tolerance():
    from boiling_point.audit import stereoisomer_conflicts
    df = pd.DataFrame({"cmpdname": ["E", "Z"], "isosmiles": ["C/C=C/CC", "C/C=C\\CC"],
                       "boiling_point_kelvin": [309.5, 310.0]})
    assert stereoisomer_conflicts(df).empty


def test_build_label_audit_notes_add_evidence_without_changing_exclusion():
    from boiling_point.audit import build_label_audit
    df = pd.DataFrame({"cmpdname": ["trans", "cis"], "isosmiles": ["OC/C=C/CO", "OC/C=C\\CO"],
                       "boiling_point_kelvin": [404.15, 508.15]})
    nist = pd.DataFrame({"cmpdname": [], "nist_tb": []})

    audit = build_label_audit(df, nist, {}, notes={"cis": "PubChem agrees"}).set_index("cmpdname")

    assert audit.loc["cis", "evidence"].endswith("; PubChem agrees")
    assert not audit["exclude"].any()
    with pytest.raises(ValueError, match="no audit check flagged"):
        build_label_audit(df, nist, {}, notes={"unknown": "x"})
