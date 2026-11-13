"""Build the frozen MoLFormer embedding cache and run the label-free probes (Step 2).

Embeds every literature compound (results/representation_study/splits.csv)
and every NIST row (nist_external.csv) from its model-input SMILES, on CPU,
with the encoder frozen and in eval mode:
- canonical: one RDKit-canonical SMILES per molecule;
- random10: the mean over 10 randomised SMILES per molecule.
Writes data/cache/molformer/ (gitignored): embeddings_{variant}.npy of shape
(14 layers, n, 768), molecules.csv (key + exact SMILES), config.json.

Probes use no boiling points (results/representation_study/):
- probes_invariance.csv: per layer, does a randomised spelling's embedding
  retrieve its own molecule, and how far does spelling move the embedding
  relative to the nearest other molecule?
- probes_chain_length.csv: per layer, within n-alkanes / 1-alcohols /
  1-amines, does embedding distance from the shortest member grow with chain
  length, and do long chains stay distinguishable (Morgan bits and counts
  shown for reference)?

Needs the optional extra: pip install -r requirements-lm.txt

Usage:
    python scripts/build_representations.py
"""
# torch first, and no xgboost fitting in this process: with xgboost's OpenMP
# runtime loaded first, MoLFormer's forward pass segfaulted (see representations.py)
import torch  # noqa: I001
import warnings

warnings.filterwarnings("ignore", message="torch was imported before boiling_point")

import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from boiling_point import representations as rep  # noqa: E402

SPLITS_CSV = "results/representation_study/splits.csv"
NIST_CSV = "results/representation_study/nist_external.csv"
CACHE_DIR = "data/cache/molformer"
OUT_DIR = "results/representation_study"
N_RANDOM, RANDOM_SEED, N_THREADS, DEVICE = 10, 0, 4, "cpu"
SERIES = {"n-alkane": "", "1-alcohol": "O", "1-amine": "N"}   # chain of k carbons + this end group
LONG_CHAIN = 8   # beyond C8, binary Morgan bits stop changing for n-alkanes


def molecules() -> pd.DataFrame:
    lit = pd.read_csv(SPLITS_CSV)
    nist = pd.read_csv(NIST_CSV)
    out = pd.concat([pd.DataFrame({"key": "literature:" + lit["mol_id"].astype(str), "isosmiles": lit["isosmiles"]}),
                     pd.DataFrame({"key": "nist:" + nist.index.astype(str), "isosmiles": nist["isosmiles"]})],
                    ignore_index=True)
    out["smiles"] = out["isosmiles"].map(rep.model_smiles)
    return out


def chain_length_probe(unique_smiles, canonical) -> pd.DataFrame:
    """Per series and representation: Spearman(chain length, distance from the
    shortest member), and median distance between consecutive homologues up to
    and beyond C8."""
    index = {s: i for i, s in enumerate(unique_smiles)}
    bits_gen = rdFingerprintGenerator.GetMorganGenerator(radius=rep.MORGAN_RADIUS, fpSize=rep.MORGAN_BITS)
    rows = []
    for series, end in SERIES.items():
        members = [(k, "C" * k + end) for k in range(1, 60) if "C" * k + end in index]
        lengths = np.array([k for k, _ in members])
        smiles = [s for _, s in members]
        reps = {name: canonical[layer, [index[s] for s in smiles]] for layer, name in enumerate(rep.LAYER_NAMES)}
        reps["morgan_bits"] = np.array([bits_gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles],
                                       dtype=float)
        reps["morgan_counts"] = rep.morgan_counts(smiles)
        for name, x in reps.items():
            from_first = rep.cosine_distance(x, np.repeat(x[:1], len(x), axis=0))
            consecutive = rep.cosine_distance(x[1:], x[:-1])
            long = lengths[1:] > LONG_CHAIN
            rows.append({"series": series, "representation": name, "n": len(x),
                         "chain_lengths": f"C{lengths.min()}-C{lengths.max()}",
                         "spearman_distance_vs_length": spearmanr(lengths, from_first).statistic,
                         f"median_step_distance_upto_C{LONG_CHAIN}": float(np.median(consecutive[~long])),
                         f"median_step_distance_beyond_C{LONG_CHAIN}": float(np.median(consecutive[long])),
                         f"share_steps_beyond_C{LONG_CHAIN}_unchanged": float(np.mean(consecutive[long] < 1e-9))})
    return pd.DataFrame(rows)


def main():
    mols = molecules()
    unique = list(dict.fromkeys(mols["smiles"]))
    lit_unique = list(dict.fromkeys(mols.loc[mols["key"].str.startswith("literature:"), "smiles"]))
    print(f"{len(mols)} molecules, {len(unique)} distinct model-input SMILES ({len(lit_unique)} literature)")

    randomized = [rep.randomized_smiles(s, N_RANDOM, RANDOM_SEED) for s in unique]
    for s, variants in zip(unique, randomized):
        if {rep.model_smiles(v) for v in variants} != {s}:
            raise ValueError(f"a randomised SMILES of {s} is a different molecule")

    model, tokenizer = rep.load_molformer(DEVICE, n_threads=N_THREADS)
    start = time.time()
    canonical = rep.embed(unique, model, tokenizer)
    random_all = rep.embed([v for vs in randomized for v in vs], model, tokenizer)
    random_all = random_all.reshape(len(rep.LAYER_NAMES), len(unique), N_RANDOM, -1)
    print(f"embedded {len(unique) * (1 + N_RANDOM)} SMILES in {time.time() - start:.0f}s")

    position = {s: i for i, s in enumerate(unique)}
    rows = mols["smiles"].map(position).to_numpy()
    config = rep.cache_config(N_RANDOM, RANDOM_SEED, DEVICE, N_THREADS)
    rep.save_cache(CACHE_DIR, mols["key"], mols["smiles"],
                   {"canonical": canonical[:, rows], "random10": random_all.mean(axis=2)[:, rows]}, config)
    print(f"cache written to {CACHE_DIR}")

    lit_idx = [position[s] for s in lit_unique]
    invariance = pd.DataFrame([{"layer": name, **rep.retrieval_invariance(canonical[layer, lit_idx],
                                                                           random_all[layer, lit_idx])}
                               for layer, name in enumerate(rep.LAYER_NAMES)])
    invariance.to_csv(f"{OUT_DIR}/probes_invariance.csv", index=False)
    chain = chain_length_probe(lit_unique, canonical[:, lit_idx])
    chain.to_csv(f"{OUT_DIR}/probes_chain_length.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(invariance.to_string(index=False))
        print(chain.to_string(index=False))


if __name__ == "__main__":
    main()
