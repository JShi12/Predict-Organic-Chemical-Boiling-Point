"""Separating encoder, head and layer in C5 (representation study, exploratory).

C5 compares fine-tuned MoLFormer (MLP head on the pooled output, encoder
trained) with frozen MoLFormer + Ridge (layer chosen in the inner CV: layer 1
in 13 of 15 folds). This trains the fine-tuning head on *frozen* canonical
embeddings at hidden_1 and the pooled output, on the fine-tuning runs' own fit
and stopping rows, with two recipes:
  R1  the fine-tuning recipe: rates 1e-5, 3e-5, 1e-4; <= 30 epochs, patience 5
  R2  room to converge: rates 1e-4, 3e-4, 1e-3, 3e-3; <= 300 epochs, patience 20
(seed 0 at each rate, the rate with the lowest mean of its best 3 stopping
MAEs, then seeds 1 and 2), and refits Ridge with the layer probe's alpha on
the same fit rows. Paired contrasts over the 15 familiar folds use the
corrected resampled t-test (raw p, no Holm). C5 is unchanged.

Writes results/representation_study/finetune/head_control/. No xgboost model
is fitted here (torch shares the process).

    python scripts/run_head_control.py [--smoke] [--n-jobs 8]
"""
import argparse
import os
import sys
import tempfile
import warnings

sys.path.insert(0, "scripts")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import t as t_dist  # noqa: E402
from sklearn.base import clone  # noqa: E402

from boiling_point import representations as rep, study, study_stats  # noqa: E402  (xgboost before torch)
from run_finetuning import CACHE_DIR, load, tasks  # noqa: E402

FT = "results/representation_study/finetune"
PROBE = "results/representation_study/frozen/layer_probe.csv"
OUT = f"{FT}/head_control"
LAYERS = {"hidden_1": 1, "pooler": 13}
RECIPES = {"R1": {"lrs": (1e-5, 3e-5, 1e-4), "max_epochs": 30, "patience": 5},
           "R2": {"lrs": (1e-4, 3e-4, 1e-3, 3e-3), "max_epochs": 300, "patience": 20}}
CONTRASTS = [("head at the pooled output (R1)", "mlp_pooler_R1", "ridge_pooler_matched"),
             ("head at the pooled output (R2)", "mlp_pooler_R2", "ridge_pooler_matched"),
             ("encoder training (R1)", "finetuned", "mlp_pooler_R1"),
             ("encoder training (R2)", "finetuned", "mlp_pooler_R2"),
             ("head at layer 1 (R1)", "mlp_hidden_1_R1", "ridge_hidden_1_matched"),
             ("head at layer 1 (R2)", "mlp_hidden_1_R2", "ridge_hidden_1_matched"),
             ("layer with an MLP head (R1)", "mlp_hidden_1_R1", "mlp_pooler_R1"),
             ("layer with an MLP head (R2)", "mlp_hidden_1_R2", "mlp_pooler_R2")]


def head_runs(task, layer_name, recipe, X, y, smoke):
    """All runs of one (task, layer, recipe): seed 0 per rate, then seeds 1-2 at the chosen rate."""
    import torch
    torch.set_num_threads(1)
    from boiling_point import finetune
    cfg = RECIPES[recipe]
    lrs = cfg["lrs"][:1] if smoke else cfg["lrs"]
    Xl = X[LAYERS[layer_name]]
    fit, stop, test = task["fit"], task["stop"], task["test"]

    def run(lr, seed):
        r = finetune.train_head(Xl[fit], y[fit], Xl[stop], y[stop], Xl[test], lr, seed,
                                max_epochs=2 if smoke else cfg["max_epochs"], patience=cfg["patience"])
        return {"task": task["task"], "layer": layer_name, "recipe": recipe, "lr": lr, "seed": seed,
                "best_epoch": r["best_epoch"], "epochs_run": len(r["log"]), "best_stop_mae": r["best_stop_mae"],
                "stop_mae_best3": r["stop_mae_best3"], "seconds": r["seconds"], "test": r["test"]}

    first = [run(lr, 0) for lr in lrs]
    chosen = min(first, key=lambda r: r["stop_mae_best3"])["lr"]
    runs = first + ([] if smoke else [run(chosen, s) for s in (1, 2)])
    for r in runs:
        r["chosen_lr"] = r["lr"] == chosen
    return runs


def ridge_matched(d, familiar, X, probe):
    """Ridge with the probe's alpha per (task, layer): refitted on the full
    training fold (must reproduce the probe) and on the fit rows (matched)."""
    rows, checks = [], []
    for t in familiar:
        for name, layer in LAYERS.items():
            pr = probe.loc[(t["task"], layer)]
            est = clone(study.ridge_pipeline("molformer")).set_params(regressor__model__alpha=pr["alpha"])
            full = est.fit(X[layer][t["train"]], d["y"][t["train"]]).predict(X[layer][t["test"]])
            checks.append(abs(np.mean(np.abs(full - d["y"][t["test"]])) - pr["test_mae"]))
            est = clone(study.ridge_pipeline("molformer")).set_params(regressor__model__alpha=pr["alpha"])
            pred = est.fit(X[layer][t["fit"]], d["y"][t["fit"]]).predict(X[layer][t["test"]])
            rows.append(pd.DataFrame({"task": t["task"], "method": f"ridge_{name}_matched", "seed": 0,
                                      "key": d["keys"][t["test"]], "y_true": d["y"][t["test"]], "y_pred": pred}))
    if max(checks) > 1e-6:
        raise ValueError(f"Ridge refit doesn't reproduce the layer probe (max diff {max(checks):.2e} K)")
    return pd.concat(rows, ignore_index=True), max(checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--smoke", action="store_true", help="1 fold, 1 rate, 1 seed, 2 epochs, temporary output")
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    out = tempfile.mkdtemp() if args.smoke else OUT

    d = load()
    familiar = [t for t in tasks(d) if t["regime"] == "familiar"]
    if args.smoke:
        familiar = familiar[:1]
    X = np.asarray(rep.load_cache(CACHE_DIR, d["keys"], d["smiles"], "canonical", rep.checked_config()),
                   dtype=np.float32)
    probe = pd.read_csv(PROBE).query("regime == 'familiar' and representation == 'molformer'") \
        .set_index(["task", "layer"])

    jobs = [(t, name, recipe) for t in familiar for name in LAYERS for recipe in RECIPES]
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(head_runs)(t, name, recipe, X, d["y"], args.smoke) for t, name, recipe in jobs)
    by_task = {t["task"]: t for t in familiar}
    runs, preds = [], []
    for group in results:
        for r in group:
            runs.append({k: v for k, v in r.items() if k != "test"})
            if r["chosen_lr"]:
                t = by_task[r["task"]]
                preds.append(pd.DataFrame({"task": r["task"], "method": f"mlp_{r['layer']}_{r['recipe']}",
                                           "seed": r["seed"], "key": d["keys"][t["test"]],
                                           "y_true": d["y"][t["test"]], "y_pred": r["test"]}))
    ridge, check = ridge_matched(d, familiar, X.astype(np.float64), probe)
    preds = pd.concat(preds + [ridge], ignore_index=True)

    # per fold: mean single-seed MAE (Ridge has one "seed")
    fold = preds.assign(ae=(preds["y_pred"] - preds["y_true"]).abs()).groupby(["task", "method", "seed"])["ae"] \
        .mean().groupby(["task", "method"]).mean().unstack("method")
    c5 = pd.read_csv(f"{FT}/c5_folds.csv").set_index("task")
    fold["finetuned"] = c5["mean"]
    fold["frozen_best"] = c5["frozen_mae"]
    for name, layer in LAYERS.items():
        fold[f"ridge_{name}_full"] = probe.xs(layer, level="layer")["test_mae"]
    fold = fold.loc[[t["task"] for t in familiar]]

    n_train = float(np.mean([len(t["fit"]) for t in familiar]))
    n_test = float(np.mean([len(t["test"]) for t in familiar]))
    rows = []
    for label, a, b in CONTRASTS:
        if a not in fold or b not in fold or len(fold) < 2:
            continue
        r = study_stats.corrected_resampled_ttest(fold[a] - fold[b], n_train, n_test)
        half = t_dist.ppf(0.975, r["df"]) * r["se"]
        rows.append({"contrast": label, "first": a, "second": b, "estimate": r["estimate"], "se": r["se"],
                     "ci_low": r["estimate"] - half, "ci_high": r["estimate"] + half, "p_raw": r["p"],
                     "folds_first_better": int((fold[a] < fold[b]).sum()), "n_folds": len(fold)})
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(runs).to_csv(f"{out}/runs.csv", index=False)
    preds.to_csv(f"{out}/predictions.csv", index=False)
    fold.to_csv(f"{out}/folds.csv")
    pd.DataFrame(rows).to_csv(f"{out}/contrasts.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"Ridge refit on the full fold reproduces the layer probe (max diff {check:.1e} K); output {out}")
    print(fold.agg(["mean", "std"]).T.round(2).to_string())
    runs_df = pd.DataFrame(runs)
    kept = runs_df[runs_df["chosen_lr"]].assign(
        at_limit=lambda r: r["epochs_run"] == r["recipe"].map({k: v["max_epochs"] for k, v in RECIPES.items()}))
    print(kept.groupby(["layer", "recipe"]).agg(
        lr=("lr", lambda s: s.value_counts().to_dict()), best_epoch=("best_epoch", "median"),
        epochs_run=("epochs_run", "median"), runs_at_epoch_limit=("at_limit", "sum"),
        seconds=("seconds", "median")).to_string())
    if rows:
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
