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
