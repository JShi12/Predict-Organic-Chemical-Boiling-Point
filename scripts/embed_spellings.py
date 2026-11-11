"""Per-spelling frozen MoLFormer embeddings for the spelling-sensitivity probe
(representation study, Step 7).

The Step 2 cache keeps only the mean over 10 randomised SMILES per molecule.
This re-embeds the same spellings (same seed, same order and batches as
scripts/build_representations.py), checks that their mean reproduces the
cached `random10` embeddings, and keeps each spelling for the literature
molecules at the layers the familiar-regime Ridge models chose.

Writes data/cache/molformer/spellings/ (gitignored): embeddings.npy of shape
(n_layers kept, n literature, 10, 768), layers.json, molecules.csv (key, model
SMILES, and the 10 spellings).

Needs the optional extra: pip install -r requirements-lm.txt

    python scripts/embed_spellings.py
"""
# torch first, and no xgboost in this process (see representations.py)
import torch  # noqa: F401, I001
import warnings

warnings.filterwarnings("ignore", message="torch was imported before boiling_point")

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, "scripts")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from boiling_point import representations as rep  # noqa: E402
from build_representations import CACHE_DIR, DEVICE, N_RANDOM, N_THREADS, RANDOM_SEED, molecules  # noqa: E402

OUT = f"{CACHE_DIR}/spellings"
SELECTION = "results/representation_study/frozen/ridge_selection.csv"


def main():
    sel = pd.read_csv(SELECTION)
    layers = sorted(int(x) for x in sel.loc[(sel["representation"] == "molformer")
                                            & (sel["regime"] == "familiar"), "layer"].unique())
    mols = molecules()
    unique = list(dict.fromkeys(mols["smiles"]))
    randomized = [rep.randomized_smiles(s, N_RANDOM, RANDOM_SEED) for s in unique]
    print(f"{len(unique)} distinct SMILES; layers kept: {[rep.LAYER_NAMES[i] for i in layers]}", flush=True)

    model, tokenizer = rep.load_molformer(DEVICE, n_threads=N_THREADS)
    start = time.time()
    random_all = rep.embed([v for vs in randomized for v in vs], model, tokenizer)
    random_all = random_all.reshape(len(rep.LAYER_NAMES), len(unique), N_RANDOM, -1)
    print(f"embedded {len(unique) * N_RANDOM} SMILES in {time.time() - start:.0f}s", flush=True)

    # check against the cached 10-spelling mean
    position = {s: i for i, s in enumerate(unique)}
    rows = mols["smiles"].map(position).to_numpy()
    cached = rep.load_cache(CACHE_DIR, mols["key"], mols["smiles"], "random10",
                            rep.checked_config(N_RANDOM, RANDOM_SEED))
    diff = float(np.abs(random_all.mean(axis=2)[:, rows] - cached).max())
    print(f"max |mean of spellings - cached random10| = {diff:.2e}")
    if diff > 1e-5:
        raise ValueError("per-spelling embeddings do not reproduce the cached mean")

    lit = mols["key"].str.startswith("literature:").to_numpy()
    lit_rows = rows[lit]
    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/embeddings.npy", random_all[layers][:, lit_rows])
    table = mols.loc[lit, ["key", "smiles"]].reset_index(drop=True)
    for j in range(N_RANDOM):
        table[f"spelling_{j}"] = [randomized[i][j] for i in lit_rows]
    table.to_csv(f"{OUT}/molecules.csv", index=False)
    with open(f"{OUT}/layers.json", "w") as f:
        json.dump({"layers": layers, "names": [rep.LAYER_NAMES[i] for i in layers],
                   "config": rep.cache_config(N_RANDOM, RANDOM_SEED, DEVICE, N_THREADS),
                   "max_abs_diff_vs_random10": diff}, f, indent=1)
    print(f"written {OUT}: {random_all[layers][:, lit_rows].shape}")


if __name__ == "__main__":
    main()
