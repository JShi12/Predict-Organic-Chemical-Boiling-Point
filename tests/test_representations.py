import json

import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem

from boiling_point import representations as rep


@pytest.mark.parametrize("smiles, expected", [
    ("[2H]C([2H])([2H])[2H]", "C"),                  # isotopes removed as atoms, not written as [H]
    ("C[C@@H](O)[C@@H](C)O", "CC(O)C(C)O"),          # chirality removed
    ("OC/C=C\\CO", "OCC=CCO"),                       # double-bond geometry removed
])
def test_model_smiles_strips_stereochemistry_and_isotopes(smiles, expected):
    assert rep.model_smiles(smiles) == expected


def test_morgan_counts_see_chain_length_and_are_deterministic():
    x = rep.morgan_counts(["CCCCCCCC", "CCCCCCCCCC"])
    assert x.shape == (2, rep.MORGAN_BITS)
    assert not (x[0] == x[1]).all()                   # binary bits would be identical here
    assert (rep.morgan_counts(["CCCCCCCC"]) == x[:1]).all()


def test_curated_log_from_model_smiles_matches_isomeric_input_except_isotopes():
    iso = ["C[C@@H](O)[C@@H](C)O", "OC/C=C\\CO", "[2H]C([2H])([2H])[2H]"]
    a = rep.curated_log(iso).to_numpy()
    b = rep.curated_log([rep.model_smiles(s) for s in iso]).to_numpy()
    assert np.allclose(a[:2], b[:2])                  # stereochemistry doesn't affect the descriptors
    assert not np.allclose(a[2], b[2])                # explicit deuterium atoms did


def test_randomized_smiles_are_reproducible_valid_and_order_independent():
    s = "CC(O)C(C)CCN"
    a = rep.randomized_smiles(s, 5, seed=0)
    assert a == rep.randomized_smiles(s, 5, seed=0)
    assert a != rep.randomized_smiles(s, 5, seed=1)
    assert {rep.model_smiles(v) for v in a} == {rep.model_smiles(s)}
    rep.randomized_smiles("CCO", 5, seed=0)           # other molecules in between don't change the draw
    assert a == rep.randomized_smiles(s, 5, seed=0)


def test_masked_mean_ignores_padding():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(rep.masked_mean(hidden, mask), torch.tensor([[2.0, 3.0]]))


def _config():
    return {k: v for k, v in {"model_id": "m", "revision": "r", "code_sha256": {"a": "1"},
                              "smiles_prep_version": "1", "pooling": "mean", "layers": ["l0", "l1"],
                              "n_random": 10, "random_seed": 0}.items()}


def test_cache_round_trip_returns_rows_in_the_requested_order(tmp_path):
    arr = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    rep.save_cache(str(tmp_path), ["a", "b", "c"], ["C", "CC", "CCC"], {"canonical": arr}, _config())

    out = rep.load_cache(str(tmp_path), ["c", "a"], ["CCC", "C"], "canonical", _config())

    assert np.array_equal(out, arr[:, [2, 0]])


@pytest.mark.parametrize("keys, smiles, config_change, error", [
    (["d"], ["C"], {}, KeyError),                                # molecule not cached
    (["a"], ["CC"], {}, ValueError),                             # SMILES changed since caching
    (["a"], ["C"], {"revision": "other"}, ValueError),           # stale model revision
    (["a"], ["C"], {"smiles_prep_version": "2"}, ValueError),    # SMILES preparation changed
])
def test_cache_refuses_missing_changed_or_stale_entries(tmp_path, keys, smiles, config_change, error):
    rep.save_cache(str(tmp_path), ["a", "b"], ["C", "CC"], {"canonical": np.zeros((2, 2, 3), np.float32)},
                   _config())
    with pytest.raises(error):
        rep.load_cache(str(tmp_path), keys, smiles, "canonical", {**_config(), **config_change})


def test_cache_config_check_ignores_unchecked_fields(tmp_path):
    rep.save_cache(str(tmp_path), ["a"], ["C"], {"canonical": np.zeros((2, 1, 3), np.float32)},
                   {**_config(), "created": "2026-11-07"})
    rep.load_cache(str(tmp_path), ["a"], ["C"], "canonical", {**_config(), "created": "2030-01-01"})
    assert json.load(open(tmp_path / "config.json"))["created"] == "2026-11-07"


def test_retrieval_invariance_separates_robust_from_spelling_sensitive_embeddings():
    rng = np.random.default_rng(0)
    canonical = rng.normal(size=(20, 8))
    robust = canonical[:, None, :] + 0.01 * rng.normal(size=(20, 5, 8))
    sensitive = rng.normal(size=(20, 5, 8))              # spelling changes everything

    good, bad = rep.retrieval_invariance(canonical, robust), rep.retrieval_invariance(canonical, sensitive)

    assert good["retrieval_share"] == 1.0 and good["median_distance_ratio"] < 0.1
    assert bad["retrieval_share"] < 0.3 and bad["median_distance_ratio"] > 1


# --- tests that need the encoder ---------------------------------------------------------
# They run in a subprocess that imports torch first, because the encoder can't share a
# process with xgboost (see representations.py), and pytest's process has xgboost loaded.
# Skipped without the lm extra or without the weights in the local Hugging Face cache.

ENCODER_CHECKS = r"""
import json, torch, warnings
warnings.filterwarnings("ignore")
import numpy as np
from boiling_point import representations as rep
model, tok = rep.load_molformer("cpu", local_files_only=True)
alone = rep.embed(["CCO"], model, tok)
padded = rep.embed(["CCO", "CCCCCCCCCCCCCCCCCCCCO"], model, tok)
batch = tok(["CCN(CC)CC", "OCC(O)CO"], padding=True, return_tensors="pt")
with torch.no_grad():
    out = model(**batch, output_hidden_states=True)
ours = rep.masked_mean(model.LayerNorm(out.hidden_states[-1]), batch["attention_mask"])
again = rep.load_molformer("cpu", local_files_only=True)
print(json.dumps({
    "shape": list(alone.shape),
    "padding_max_diff": float(np.abs(alone[:, 0] - padded[:, 0]).max()),
    "pooler_max_diff": float((ours - out.pooler_output).abs().max()),
    "reload_identical": bool(np.array_equal(rep.embed(["CCO", "ClCCBr"], model, tok),
                                            rep.embed(["CCO", "ClCCBr"], *again))),
}))
"""


@pytest.fixture(scope="module")
def encoder_checks():
    import subprocess
    import sys
    pytest.importorskip("transformers")
    from huggingface_hub import try_to_load_from_cache
    if not isinstance(try_to_load_from_cache(rep.MOLFORMER_ID, "model.safetensors",
                                             revision=rep.MOLFORMER_REVISION), str):
        pytest.skip("MoLFormer weights not in the local Hugging Face cache")
    result = subprocess.run([sys.executable, "-c", ENCODER_CHECKS], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_embedding_does_not_depend_on_padding(encoder_checks):
    assert encoder_checks["shape"] == [len(rep.LAYER_NAMES), 1, 768]
    assert encoder_checks["padding_max_diff"] < 1e-5


def test_last_layer_pooling_matches_the_models_own_pooler(encoder_checks):
    assert encoder_checks["pooler_max_diff"] < 1e-5


def test_reloading_gives_identical_embeddings(encoder_checks):
    assert encoder_checks["reload_identical"]
