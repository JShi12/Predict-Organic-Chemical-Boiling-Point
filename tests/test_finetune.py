import json

import numpy as np
import pandas as pd
import pytest

from boiling_point import representations as rep, splits


def _table(n_groups=300, seed=0):
    rng = np.random.default_rng(seed)
    fams = rng.choice(["alkane", "alkene/alkyne", "alcohol/phenol", "amine"], size=n_groups, p=[.3, .3, .3, .1])
    reps = rng.choice([1, 1, 2], size=n_groups)
    return pd.DataFrame({"family": np.repeat(fams, reps), "group": np.repeat(np.arange(n_groups), reps)})


def test_validation_split_is_grouped_stratified_and_disjoint():
    df = _table()
    train = np.arange(0, len(df), 1)[: int(0.8 * len(df))]
    fit, stop, cal = splits.validation_split(train, df["family"].to_numpy(), df["group"].to_numpy(), 0.15, seed=0)

    parts = [set(fit), set(stop), set(cal)]
    assert set().union(*parts) == set(train)                                   # nothing lost, nothing added
    assert sum(len(p) for p in parts) == len(train)                            # disjoint
    g = df["group"].to_numpy()
    assert not set(g[stop]) & set(g[fit]) and not set(g[cal]) & set(g[fit]) and not set(g[stop]) & set(g[cal])
    assert abs(len(stop) + len(cal) - 0.15 * len(train)) <= 6                  # ~15%, whole groups
    assert abs(len(stop) - len(cal)) <= 6                                      # halved
    held = df.iloc[np.concatenate([stop, cal])]["family"].value_counts(normalize=True)
    whole = df.iloc[train]["family"].value_counts(normalize=True)
    assert (held - whole).abs().max() < 0.06                                   # family shares kept


def test_validation_split_is_reproducible_and_seed_dependent():
    df = _table()
    train = np.arange(len(df))
    a = splits.validation_split(train, df["family"], df["group"], seed=0)
    b = splits.validation_split(train, df["family"], df["group"], seed=0)
    c = splits.validation_split(train, df["family"], df["group"], seed=1)
    assert all((x == y).all() for x, y in zip(a, b))
    assert not (a[1].tolist() == c[1].tolist())


def test_epoch_smiles_are_valid_reproducible_and_change_between_epochs():
    from boiling_point import finetune
    smiles = ["CC(O)C(C)CCN", "CCCCCCO", "CCN(CC)CC"]
    e0 = finetune.epoch_smiles(smiles, seed=0, epoch=0)
    assert e0 == finetune.epoch_smiles(smiles, seed=0, epoch=0)
    assert e0 != finetune.epoch_smiles(smiles, seed=0, epoch=1)
    assert [rep.model_smiles(s) for s in e0] == [rep.model_smiles(s) for s in smiles]


def test_head_maps_pooled_embeddings_to_one_value():
    import torch
    from boiling_point import finetune
    assert finetune.make_head()(torch.zeros(4, 768)).shape == (4, 1)


# --- encoder-dependent checks, in a subprocess (the encoder can't share pytest's process with xgboost) ---

TRAINING_CHECKS = r"""
import json, torch, warnings
warnings.filterwarnings("ignore")
from boiling_point import finetune
fit = ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCN", "CCCCN", "CCCC", "CCCCC"]
y = [351.4, 370.3, 390.8, 411.0, 321.6, 351.0, 272.7, 309.2]
args = (fit, y, ["CCCCCCO", "CCCCCC"], [430.0, 341.9], ["CCCCCCCO"], ["CCCCCCC"])
a = finetune.train_run(*args, lr=3e-5, seed=0, max_epochs=2, tta=False)
b = finetune.train_run(*args, lr=3e-5, seed=0, max_epochs=2, tta=False)
print(json.dumps({"loss_a": [r["train_loss"] for r in a["log"]], "loss_b": [r["train_loss"] for r in b["log"]],
                  "pred_diff": float(abs(a["test_canonical"][0] - b["test_canonical"][0]))}))
"""


@pytest.fixture(scope="module")
def training_checks():
    import subprocess
    import sys
    pytest.importorskip("transformers")
    from huggingface_hub import try_to_load_from_cache
    if not isinstance(try_to_load_from_cache(rep.MOLFORMER_ID, "model.safetensors",
                                             revision=rep.MOLFORMER_REVISION), str):
        pytest.skip("MoLFormer weights not in the local Hugging Face cache")
    result = subprocess.run([sys.executable, "-c", TRAINING_CHECKS], capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]   # train_run raises if the random features changed
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_training_keeps_random_features_and_is_seeded(training_checks):
    assert len(training_checks["loss_a"]) == 2
    assert training_checks["loss_a"] == pytest.approx(training_checks["loss_b"], rel=1e-5)
    assert training_checks["pred_diff"] < 1e-3


def test_run_path_handles_family_names_with_a_slash(tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    import run_finetuning
    path = run_finetuning.run_path(str(tmp_path), "alcohol/phenol", 1e-5, 0)
    assert path.startswith(f"{tmp_path}/runs/") and "/" not in path[len(f"{tmp_path}/runs/"):]


def _head_data(seed=0, n=300, d=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = 400 + 30 * X[:, 0] - 20 * np.tanh(X[:, 1]) + rng.normal(0, 1, n)
    return X[:200], y[:200], X[200:250], y[200:250], X[250:], y[250:]


def test_train_head_learns_and_beats_the_mean():
    pytest.importorskip("torch")
    from boiling_point import finetune
    Xf, yf, Xs, ys, Xt, yt = _head_data()
    r = finetune.train_head(Xf, yf, Xs, ys, Xt, lr=3e-3, seed=0, max_epochs=200, patience=20)
    mae = np.mean(np.abs(r["test"] - yt))
    assert mae < 0.3 * np.mean(np.abs(yt - yf.mean()))
    assert r["best_stop_mae"] == min(e["stop_mae"] for e in r["log"])
    assert r["log"][r["best_epoch"]]["stop_mae"] == r["best_stop_mae"]


def test_train_head_is_deterministic_and_stops_early():
    pytest.importorskip("torch")
    from boiling_point import finetune
    Xf, yf, Xs, ys, Xt, _ = _head_data(1)
    a = finetune.train_head(Xf, yf, Xs, ys, Xt, lr=1e-3, seed=4, max_epochs=40, patience=3)
    b = finetune.train_head(Xf, yf, Xs, ys, Xt, lr=1e-3, seed=4, max_epochs=40, patience=3)
    assert np.array_equal(a["test"], b["test"])
    assert len(a["log"]) <= 40 and len(a["log"]) - 1 - a["best_epoch"] <= 3


def test_layer_index_parses_readouts():
    pytest.importorskip("torch")
    from boiling_point import finetune
    assert finetune.layer_index("pooled") is None
    assert finetune.layer_index("layer:1") == 1 and finetune.layer_index("layer:3") == 3
    for bad in ("layer:0", "mix", "hidden_1"):
        with pytest.raises(ValueError):
            finetune.layer_index(bad)


READOUT_CHECKS = r"""
import torch  # first, as in the encoder scripts
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from boiling_point import finetune, representations as rep
smiles = ["CCO", "CCCCCCO", "CC(C)CN", "C=CCCC", "OCC(O)CO"]
model, tok = rep.load_molformer("cpu", local_files_only=True, n_threads=2)
full = rep.embed(smiles, model, tok)[1]                       # cached-style hidden_1
finetune.truncate(model, "layer:1")
with torch.no_grad():
    batch = tok(smiles, padding=True, return_tensors="pt")
    trunc = finetune.encode(model, batch, "layer:1").numpy()
fit = ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCN", "CCCCN", "CCCC", "CCCCC"]
y = [351.4, 370.3, 390.8, 411.0, 321.6, 351.0, 272.7, 309.2]
args = (fit, y, ["CCCCCCO", "CCCCCC"], [430.0, 341.9], ["CCCCCCCO"], ["CCCCCCC"])
a = finetune.train_run(*args, lr=3e-5, seed=0, max_epochs=2, tta=False, readout="layer:1", head_lr=1e-3)
b = finetune.train_run(*args, lr=3e-5, seed=0, max_epochs=2, tta=False, readout="layer:1", head_lr=1e-3)
c = finetune.train_run(*args, lr=3e-5, seed=0, max_epochs=2, tta=False, readout="layer:1")
print(json.dumps({"n_blocks": len(model.encoder.layer), "max_diff": float(np.abs(full - trunc).max()),
                  "loss_a": [r["train_loss"] for r in a["log"]], "loss_b": [r["train_loss"] for r in b["log"]],
                  "loss_c": [r["train_loss"] for r in c["log"]]}))
"""


@pytest.fixture(scope="module")
def readout_checks():
    import subprocess
    import sys
    pytest.importorskip("transformers")
    from huggingface_hub import try_to_load_from_cache
    if not isinstance(try_to_load_from_cache(rep.MOLFORMER_ID, "model.safetensors",
                                             revision=rep.MOLFORMER_REVISION), str):
        pytest.skip("MoLFormer weights not in the local Hugging Face cache")
    result = subprocess.run([sys.executable, "-c", READOUT_CHECKS], capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_truncated_layer_readout_matches_the_cached_pooling(readout_checks):
    assert readout_checks["n_blocks"] == 1
    assert readout_checks["max_diff"] < 1e-5


def test_layer_readout_training_is_seeded_and_head_rate_matters(readout_checks):
    assert readout_checks["loss_a"] == pytest.approx(readout_checks["loss_b"], rel=1e-5)
    assert readout_checks["loss_a"][1] != pytest.approx(readout_checks["loss_c"][1], rel=1e-4)
