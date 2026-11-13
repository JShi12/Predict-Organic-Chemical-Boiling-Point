"""Molecular representations for the representation study.

Three representations of the same molecule, all built from one cleaned
"model-input" SMILES:
- curated RDKit descriptors (log size/shape variant, descriptors.py);
- Morgan count fingerprints (radius 2, 2048 bins);
- frozen MoLFormer-XL embeddings: masked mean over tokens of every hidden
  layer, cached on disk with the molecules and configuration they came from.

torch and transformers are imported inside the encoder functions, so the
rest of the module works without the optional `lm` extra.

Run the encoder in its own process, importing torch before boiling_point,
and don't fit xgboost there. torch ships its own OpenMP library and xgboost
uses the system one; on macOS (checked 2026-11-07) a process that loaded
xgboost's first segfaulted in MoLFormer's forward pass, and one that ran both
a forward pass and an xgboost fit segfaulted in either import order. Models
on cached embeddings need only numpy, so they run in separate processes.
"""
import datetime
import hashlib
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from . import descriptors

SMILES_PREP_VERSION = "1: stereochemistry and isotope labels removed, RemoveHs, RDKit canonical"
MORGAN_RADIUS, MORGAN_BITS = 2, 2048

MOLFORMER_ID = "ibm-research/MoLFormer-XL-both-10pct"
MOLFORMER_REVISION = "361063d0ad524ef77cf39b08469f6be770dc550f"
# SHA-256 of the custom model code reviewed before it was first run (2026-11-07)
MOLFORMER_CODE_SHA256 = {
    "modeling_molformer.py": "6f1ef72022de2c69e95661899422a7bb39a40a2cc5a6cb6216f14e9b7d84559c",
    "configuration_molformer.py": "b88ea8d4b7b5e54f4f186cc7a230eff308928020a030f153a038fd12c05e3bed",
}
# hidden_0 = token embeddings (no context); hidden_1..12 = encoder layers;
# pooler = the model's own pooled output: hidden_12 after the final LayerNorm, masked mean
LAYER_NAMES = [f"hidden_{i}" for i in range(13)] + ["pooler"]
POOLING = "masked mean over tokens incl. <bos>/<eos>, excl. padding"
# Configuration fields a cache must match before it is used
CHECKED_CONFIG_KEYS = ("model_id", "revision", "code_sha256", "smiles_prep_version", "pooling", "layers",
                       "n_random", "random_seed")


# --- inputs ---------------------------------------------------------------

def model_smiles(smiles: str) -> str:
    """Canonical SMILES with stereochemistry and isotope labels removed.
    Unlike isomericSmiles=False, which writes deuterium as explicit [H]
    atoms ([2H]C([2H])([2H])[2H] -> [H]C([H])([H])[H]), this edits the atoms,
    so deuterated methane becomes 'C'."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def randomized_smiles(smiles: str, n: int, seed: int = 0) -> list:
    """n valid non-canonical SMILES for one molecule, from random atom orders.
    The generator is seeded from the seed and the SMILES itself, so the result
    doesn't depend on which other molecules are processed or in what order.
    (RDKit's MolToRandomSmilesVect with randomSeed was not reproducible across
    calls when checked.)"""
    mol = Chem.MolFromSmiles(smiles)
    digest = int.from_bytes(hashlib.sha256(smiles.encode()).digest()[:8], "little")
    rng = np.random.default_rng([seed, digest])
    out = []
    for _ in range(n):
        order = [int(i) for i in rng.permutation(mol.GetNumAtoms())]
        out.append(Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False))
    return out


# --- conventional representations ------------------------------------------

def curated_log(smiles) -> pd.DataFrame:
    """Curated RDKit descriptors with log-transformed size/shape terms (the
    feature study's 'curated (log)' set, without family)."""
    return descriptors.log_size_shape(descriptors.curated_descriptors(pd.Series(list(smiles))))


def morgan_counts(smiles, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_BITS) -> np.ndarray:
    """(n, n_bits) Morgan count fingerprints. Counts, not bits: binary bits are
    identical for n-octane and longer n-alkanes."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=float)


# --- MoLFormer ---------------------------------------------------------------

def masked_mean(hidden, mask):
    """Mean over the token axis of a (batch, tokens, dim) tensor, counting only
    positions where mask (batch, tokens) is 1."""
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def _verify_code(local_files_only: bool):
    from huggingface_hub import hf_hub_download
    for name, expected in MOLFORMER_CODE_SHA256.items():
        path = hf_hub_download(MOLFORMER_ID, name, revision=MOLFORMER_REVISION, local_files_only=local_files_only)
        with open(path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"{name} at {MOLFORMER_REVISION} differs from the reviewed copy ({actual})")


def load_molformer(device: str = "cpu", local_files_only: bool = False, n_threads: int = 4):
    """(model, tokenizer) at the pinned revision, frozen and in eval mode.
    The custom model code is checked against the reviewed hashes before it is
    imported. deterministic_eval keeps the linear attention's random features
    fixed in eval mode; they are stored in the checkpoint.

    Sets torch's CPU thread count, which is recorded in the cache config.
    Call it in a process that imported torch before boiling_point and fits
    no xgboost model (see the module docstring)."""
    import torch
    from transformers import AutoModel, PreTrainedTokenizerFast

    torch.set_num_threads(n_threads)
    _verify_code(local_files_only)
    torch.manual_seed(0)
    model = AutoModel.from_pretrained(MOLFORMER_ID, revision=MOLFORMER_REVISION, code_revision=MOLFORMER_REVISION,
                                      deterministic_eval=True, trust_remote_code=True,
                                      local_files_only=local_files_only)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MOLFORMER_ID, revision=MOLFORMER_REVISION,
                                                        local_files_only=local_files_only)
    return model, tokenizer


def embed(smiles, model, tokenizer, batch_size: int = 64) -> np.ndarray:
    """(n_layers, n, hidden) float32 embeddings: masked mean of each hidden
    state, plus the model's pooled output (see LAYER_NAMES)."""
    import torch
    if model.training:
        raise ValueError("model must be in eval mode")
    smiles = list(smiles)
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch_size):
            batch = tokenizer(smiles[i:i + batch_size], padding=True, return_tensors="pt").to(device)
            result = model(**batch, output_hidden_states=True)
            mask = batch["attention_mask"]
            layers = [masked_mean(h, mask) for h in result.hidden_states] + [result.pooler_output]
            out.append(torch.stack(layers).cpu().numpy())
    return np.concatenate(out, axis=1).astype(np.float32)


# --- cache -------------------------------------------------------------------

def checked_config(n_random: int = 10, random_seed: int = 0) -> dict:
    """The CHECKED_CONFIG_KEYS fields a cache must match. Needs no torch, so a
    process that fits xgboost can validate the cache before loading it."""
    return {"model_id": MOLFORMER_ID, "revision": MOLFORMER_REVISION, "code_sha256": MOLFORMER_CODE_SHA256,
            "smiles_prep_version": SMILES_PREP_VERSION, "pooling": POOLING, "layers": LAYER_NAMES,
            "n_random": n_random, "random_seed": random_seed}


def cache_config(n_random: int, random_seed: int, device: str, n_threads: int) -> dict:
    import torch
    import transformers
    return {**checked_config(n_random, random_seed), "device": device, "torch_threads": n_threads,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "created": datetime.date.today().isoformat()}


def save_cache(cache_dir: str, keys, smiles, arrays: dict, config: dict) -> None:
    """Write each variant's (n_layers, n, hidden) array, the molecule keys with
    the exact SMILES embedded, and the configuration."""
    os.makedirs(cache_dir, exist_ok=True)
    pd.DataFrame({"key": list(keys), "smiles": list(smiles)}).to_csv(f"{cache_dir}/molecules.csv", index=False)
    for variant, array in arrays.items():
        np.save(f"{cache_dir}/embeddings_{variant}.npy", array)
    with open(f"{cache_dir}/config.json", "w") as f:
        json.dump(config, f, indent=1)


def load_cache(cache_dir: str, keys, smiles, variant: str, expected_config: dict) -> np.ndarray:
    """(n_layers, len(keys), hidden) embeddings in the requested order. Raises
    if a key is missing, a key's SMILES differs from the one embedded, or the
    cache's configuration differs on any CHECKED_CONFIG_KEYS field."""
    with open(f"{cache_dir}/config.json") as f:
        config = json.load(f)
    stale = [k for k in CHECKED_CONFIG_KEYS if config.get(k) != expected_config.get(k)]
    if stale:
        raise ValueError(f"cache configuration differs on {stale}; rebuild with scripts/build_representations.py")
    molecules = pd.read_csv(f"{cache_dir}/molecules.csv")
    position = pd.Series(np.arange(len(molecules)), index=molecules["key"])
    keys, smiles = list(keys), list(smiles)
    missing = [k for k in keys if k not in position.index]
    if missing:
        raise KeyError(f"{len(missing)} molecules not in the cache, e.g. {missing[:3]}")
    idx = position.loc[keys].to_numpy()
    cached = molecules["smiles"].to_numpy()[idx]
    changed = [k for k, a, b in zip(keys, cached, smiles) if a != b]
    if changed:
        raise ValueError(f"SMILES differ from the cached ones for {len(changed)} molecules, e.g. {changed[:3]}")
    return np.load(f"{cache_dir}/embeddings_{variant}.npy", mmap_mode="r")[:, idx, :]


# --- label-free probes ---------------------------------------------------------

def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def retrieval_invariance(canonical: np.ndarray, randomized: np.ndarray) -> dict:
    """How well one layer's embedding identifies a molecule across spellings.

    canonical: (n, d) embeddings of n distinct molecules; randomized: (n, k, d),
    k randomised-SMILES embeddings per molecule. Returns the share of
    randomised embeddings whose nearest canonical embedding (cosine) is their
    own molecule's, and the median ratio of the spelling distance (own
    canonical <-> randomised) to the distance to the nearest *other*
    molecule's canonical embedding."""
    c = _unit(canonical.astype(np.float64))
    r = _unit(randomized.astype(np.float64))
    n, k, _ = r.shape
    sims = np.einsum("nkd,md->nkm", r, c)                      # (n, k, n)
    own = sims[np.arange(n), :, np.arange(n)]                  # (n, k)
    retrieved = sims.argmax(axis=2) == np.arange(n)[:, None]
    cc = c @ c.T
    np.fill_diagonal(cc, -np.inf)
    nearest_other = 1 - cc.max(axis=1)                         # (n,)
    ratio = (1 - own) / np.maximum(nearest_other[:, None], 1e-12)
    return {"retrieval_share": float(retrieved.mean()), "median_distance_ratio": float(np.median(ratio))}


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine distance between equal-shaped (n, d) arrays (0 for two zero rows)."""
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    both_zero = (na == 0) & (nb == 0)
    dist = 1 - (a * b).sum(axis=1) / np.maximum(na * nb, 1e-12)
    return np.where(both_zero, 0.0, dist)
