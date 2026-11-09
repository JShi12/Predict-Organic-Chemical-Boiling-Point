"""Fine-tuning MoLFormer-XL for boiling-point regression (representation study, Step 4).

All 44.4M encoder parameters train, with a small head on the model's own
pooled output (final LayerNorm, masked mean over tokens). Fixed choices:
- targets standardised on the fitting set; mean squared error loss;
- AdamW (weight decay 0.01), batch 32, linear warm-up over the first epoch
  then a constant learning rate, gradient-norm clipping at 1.0;
- each epoch, every training molecule gets one randomised SMILES (the frozen
  embeddings proved very spelling-sensitive);
- at most 30 epochs, early stopping with patience 5 on the stopping half's
  canonical-SMILES MAE, keeping the best epoch's weights;
- the linear attention's random features stay fixed at the checkpoint's
  stored values during training as well as evaluation (a design choice: the
  model card doesn't say whether pretraining redrew them).

Run in a process that imported torch before boiling_point and fits no
xgboost model (see representations.py).
"""
import time

import numpy as np
import torch
from torch import nn

from . import representations as rep
from .splits import validation_split  # noqa: F401  (re-exported)

BATCH_SIZE, MAX_EPOCHS, PATIENCE, WEIGHT_DECAY, CLIP = 32, 30, 5, 0.01, 1.0
LEARNING_RATES = (1e-5, 3e-5, 1e-4)
N_TTA = 10


def epoch_smiles(smiles, seed: int, epoch: int) -> list:
    """One randomised spelling per molecule for this seed and epoch."""
    return [rep.randomized_smiles(s, 1, seed=seed * 1000 + epoch)[0] for s in smiles]


# --- model -------------------------------------------------------------------------------

def make_head(hidden: int = 768) -> nn.Module:
    return nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))


def feature_maps(model):
    return [m for m in model.modules() if type(m).__name__ == "MolformerFeatureMap"]


def set_train_mode(model, head):
    """Dropout on, but the attention's random features fixed: with
    deterministic_eval, a feature map in eval mode never redraws them."""
    model.train()
    head.train()
    for m in feature_maps(model):
        m.eval()


def predict(model, head, tokenizer, smiles, mean: float, sd: float, batch_size: int = 64) -> np.ndarray:
    model.eval()
    head.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch_size):
            batch = tokenizer(list(smiles[i:i + batch_size]), padding=True, return_tensors="pt")
            out.append(head(model(**batch).pooler_output).squeeze(-1).numpy())
    return np.concatenate(out) * sd + mean if out else np.array([])


def predict_tta(model, head, tokenizer, smiles, mean, sd, n: int = N_TTA, seed: int = 0) -> np.ndarray:
    """Mean prediction over n randomised spellings per molecule (the same
    spellings as the frozen `random10` embeddings: seed 0)."""
    spellings = [v for s in smiles for v in rep.randomized_smiles(s, n, seed)]
    return predict(model, head, tokenizer, spellings, mean, sd).reshape(len(smiles), n).mean(axis=1)


# --- one training run ------------------------------------------------------------------------

def train_run(fit_smiles, fit_y, stop_smiles, stop_y, test_smiles, calibrate_smiles, lr: float, seed: int,
              max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE, n_threads: int = 3, tta: bool = True,
              local_files_only: bool = True) -> dict:
    """Fine-tune once and predict. All SMILES are model-input SMILES; stopping
    and the returned test/calibration predictions use the canonical spelling."""
    start = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, tokenizer = rep.load_molformer("cpu", local_files_only=local_files_only, n_threads=n_threads)
    for p in model.parameters():
        p.requires_grad_(True)
    torch.manual_seed(seed)                       # head initialisation and dropout depend on the seed only
    head = make_head()
    fit_y = np.asarray(fit_y, dtype=float)
    mean, sd = fit_y.mean(), fit_y.std()
    target = torch.tensor((fit_y - mean) / sd, dtype=torch.float32)
    params = list(model.parameters()) + list(head.parameters())
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = int(np.ceil(len(fit_smiles) / BATCH_SIZE))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: min(1.0, (step + 1) / steps_per_epoch))
    rng = np.random.default_rng(seed)
    initial_features = [m.weight.clone() for m in feature_maps(model)]

    log, best_mae, best_state, best_epoch, waited = [], np.inf, None, -1, 0
    for epoch in range(max_epochs):
        set_train_mode(model, head)
        spellings = epoch_smiles(fit_smiles, seed, epoch)
        order = rng.permutation(len(spellings))
        losses = []
        for b in range(steps_per_epoch):
            idx = order[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            batch = tokenizer([spellings[i] for i in idx], padding=True, return_tensors="pt")
            loss = nn.functional.mse_loss(head(model(**batch).pooler_output).squeeze(-1), target[idx])
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, CLIP)
            optimiser.step()
            scheduler.step()
            losses.append(loss.item())
        stop_mae = float(np.mean(np.abs(predict(model, head, tokenizer, stop_smiles, mean, sd) - np.asarray(stop_y))))
        log.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "stop_mae": stop_mae})
        if stop_mae < best_mae:
            best_mae, best_epoch, waited = stop_mae, epoch, 0
            best_state = ({k: v.detach().clone() for k, v in model.state_dict().items()},
                          {k: v.detach().clone() for k, v in head.state_dict().items()})
        else:
            waited += 1
            if waited >= patience:
                break

    if not all(torch.equal(a, m.weight) for a, m in zip(initial_features, feature_maps(model))):
        raise RuntimeError("attention random features changed during training")
    model.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    best3 = float(np.mean(sorted(r["stop_mae"] for r in log)[:3]))
    return {"log": log, "best_epoch": best_epoch, "best_stop_mae": best_mae, "stop_mae_best3": best3,
            "test_canonical": predict(model, head, tokenizer, test_smiles, mean, sd),
            "test_tta": predict_tta(model, head, tokenizer, test_smiles, mean, sd) if tta else None,
            "calibrate": predict(model, head, tokenizer, calibrate_smiles, mean, sd),
            "seconds": time.time() - start}
