"""Fine-tuning that reads an early layer (representation study, Step 7c, Stage 1; exploratory).

MoLFormer truncated after block 1, the fine-tuning head on the masked mean of
block 1's tokens (the cached "hidden_1"); trainable: embeddings, block 1, head.
Everything else as in scripts/run_finetuning.py (same fit / stop / calibration
rows, spelling augmentation, seed 0 at each rate, the rate with the lowest
mean of its best 3 stopping MAEs, then seeds 1 and 2):
  E1     one learning rate for encoder and head (1e-5, 3e-5, 1e-4), as in Step 4
  E1_2r  the head at a fixed 1e-3, the encoder at the same grid
Phase "familiar": both variants on the 15 familiar folds. Phase "transfer":
the variant with the lower mean stopping-half MAE over the familiar folds
(validation data only) on the 4 held-out families and NIST, descriptive.

    python scripts/run_finetune_readout.py --check-default      # guard: Step 4 path unchanged
    python scripts/run_finetune_readout.py --pilot              # 1 epoch, timing
    python scripts/run_finetune_readout.py --phase familiar --shard 0 --n-shards 4   (and 1, 2, 3)
    python scripts/run_finetune_readout.py --phase transfer --shard 0 --n-shards 4   (and 1, 2, 3)
    python scripts/run_finetune_readout.py --collect                                 (no torch)
"""
import argparse
import sys

ARGS = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phase", choices=["familiar", "transfer"])
    parser.add_argument("--shard", type=int)
    parser.add_argument("--n-shards", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--check-default", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    ARGS = parser.parse_args()
    if not ARGS.collect:
        import torch  # noqa: F401  (before boiling_point: see representations.py)

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "scripts")
from run_finetuning import load, tasks  # noqa: E402

FT = "results/representation_study/finetune"
OUT = "results/representation_study/finetune_readout"
VARIANTS = {"E1": {"readout": "layer:1", "head_lr": None}, "E1_2r": {"readout": "layer:1", "head_lr": 1e-3}}
CONTRASTS = [("E1 − fine-tuned (pooled)", "E1", "finetuned"),
             ("E1_2r − fine-tuned (pooled)", "E1_2r", "finetuned"),
             ("E1 − frozen layer 1 + MLP, R1", "E1", "mlp_hidden_1_R1"),
             ("E1_2r − frozen layer 1 + MLP, R2", "E1_2r", "mlp_hidden_1_R2"),
             ("E1 − frozen best", "E1", "frozen_best"),
             ("E1_2r − frozen best", "E1_2r", "frozen_best"),
             ("E1 − frozen layer 1 + Ridge, same 85%", "E1", "ridge_hidden_1_matched"),
             ("E1_2r − frozen layer 1 + Ridge, same 85%", "E1_2r", "ridge_hidden_1_matched"),
             ("E1_2r − E1 (head rate)", "E1_2r", "E1")]


def run_path(out, variant, task, lr, seed):
    return f"{out}/runs/{variant}__{task.replace('/', '-')}__lr{lr:g}__s{seed}.json"


def one_run(d, t, variant, lr, seed, out, threads, max_epochs=None):
    from boiling_point import finetune
    path = run_path(out, variant, t["task"], lr, seed)
    if os.path.exists(path):
        return json.load(open(path))
    s = d["smiles"]
    r = finetune.train_run(s[t["fit"]], d["y"][t["fit"]], s[t["stop"]], d["y"][t["stop"]], s[t["test"]],
                           s[t["calibrate"]], lr, seed, max_epochs=max_epochs or finetune.MAX_EPOCHS,
                           n_threads=threads, **VARIANTS[variant])
    record = {"variant": variant, "regime": t["regime"], "task": t["task"], "lr": lr, "seed": seed,
              **{k: r[k] for k in ("log", "best_epoch", "best_stop_mae", "stop_mae_best3", "seconds")},
              "test_keys": d["keys"][t["test"]].tolist(), "test_canonical": r["test_canonical"].tolist(),
              "test_tta": r["test_tta"].tolist(), "calibrate_keys": d["keys"][t["calibrate"]].tolist(),
              "calibrate": r["calibrate"].tolist(), "n_fit": len(t["fit"])}
    os.makedirs(f"{out}/runs", exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(record, f)
    os.replace(path + ".tmp", path)
    print(f"{variant} {t['task']} lr={lr:g} seed={seed}: best epoch {r['best_epoch']} of {len(r['log'])}, "
          f"stop MAE {r['best_stop_mae']:.2f} K, {r['seconds']:.0f}s", flush=True)
    return record


def task_runs(d, t, variant, out, threads):
    from boiling_point import finetune
    first = [one_run(d, t, variant, lr, 0, out, threads) for lr in finetune.LEARNING_RATES]
    chosen = min(first, key=lambda r: r["stop_mae_best3"])["lr"]
    for seed in (1, 2):
        one_run(d, t, variant, chosen, seed, out, threads)


def chosen_variant(out):
    """Transfer variant: lower mean stopping-half MAE (kept runs) over the familiar folds."""
    sel = selection(out)
    fam = sel[(sel["regime"] == "familiar") & sel["chosen_lr"]]
    if fam.groupby("variant")["task"].nunique().min() < 15:
        raise SystemExit("finish the familiar phase first")
    return fam.groupby("variant")["best_stop_mae"].mean().idxmin()


def selection(out):
    runs = [json.load(open(f"{out}/runs/{f}")) for f in sorted(os.listdir(f"{out}/runs")) if f.endswith(".json")]
    rows = []
    for (variant, task), g in pd.DataFrame(runs).groupby(["variant", "task"]):
        first = g[g["seed"] == 0]
        chosen = first.loc[first["stop_mae_best3"].idxmin(), "lr"]
        for _, r in g.iterrows():
            rows.append({"variant": variant, "regime": r["regime"], "task": task, "lr": r["lr"], "seed": r["seed"],
                         "chosen_lr": r["lr"] == chosen, "stop_mae_best3": r["stop_mae_best3"],
                         "best_stop_mae": r["best_stop_mae"], "best_epoch": r["best_epoch"],
                         "epochs_run": len(r["log"]), "seconds": r["seconds"], "n_fit": r["n_fit"]})
    return pd.DataFrame(rows)


def check_default(d, all_tasks, threads):
    """The default (Step 4) path must reproduce an original run's first 2 epochs."""
    from boiling_point import finetune
    t = next(t for t in all_tasks if t["task"] == "r0f0")
    s = d["smiles"]
    r = finetune.train_run(s[t["fit"]], d["y"][t["fit"]], s[t["stop"]], d["y"][t["stop"]], s[t["test"]],
                           s[t["calibrate"]], 1e-5, 0, max_epochs=2, n_threads=3, tta=False)
    log = pd.read_csv(f"{FT}/training_log.csv").query("task == 'r0f0' and lr == 1e-5 and seed == 0")
    log = log.sort_values("epoch").head(2)
    new = pd.DataFrame(r["log"])
    diff = max(np.abs(new["train_loss"].to_numpy() - log["train_loss"].to_numpy()).max(),
               np.abs(new["stop_mae"].to_numpy() - log["stop_mae"].to_numpy()).max())
    print(f"default path vs Step 4 log (r0f0, lr 1e-5, seed 0, 2 epochs): max diff {diff:.2e}")
    if diff > 1e-6:
        raise SystemExit("the default fine-tuning path no longer reproduces Step 4")


def corrected_contrasts(fold, n_train, n_test):
    from scipy.stats import t as t_dist

    from boiling_point import study_stats
    rows = []
    for label, a, b in CONTRASTS:
        if a not in fold or b not in fold:
            continue
        r = study_stats.corrected_resampled_ttest(fold[a] - fold[b], n_train, n_test)
        half = t_dist.ppf(0.975, r["df"]) * r["se"]
        rows.append({"contrast": label, "first": a, "second": b, "estimate": r["estimate"], "se": r["se"],
                     "ci_low": r["estimate"] - half, "ci_high": r["estimate"] + half, "p_raw": r["p"],
                     "folds_first_better": int((fold[a] < fold[b]).sum()), "n_folds": len(fold),
                     "at_least_1K_better": bool(r["estimate"] <= -1.0 and r["estimate"] + half < 0)})
    return pd.DataFrame(rows)


def collect(d, all_tasks, out):
    runs = [json.load(open(f"{out}/runs/{f}")) for f in sorted(os.listdir(f"{out}/runs")) if f.endswith(".json")]
    sel = selection(out)
    y_of = dict(zip(d["keys"], d["y"]))
    kept = {(r["variant"], r["task"], r["lr"], r["seed"]) for r in sel[sel["chosen_lr"]].to_dict("records")}
    preds = []
    for r in runs:
        if (r["variant"], r["task"], r["lr"], r["seed"]) not in kept:
            continue
        for mode in ("canonical", "tta"):
            preds.append(pd.DataFrame({"variant": r["variant"], "regime": r["regime"], "task": r["task"],
                                       "seed": r["seed"], "lr": r["lr"], "inference": mode, "key": r["test_keys"],
                                       "y_true": [y_of[k] for k in r["test_keys"]],
                                       "y_pred": r[f"test_{mode}"]}))
    preds = pd.concat(preds, ignore_index=True)
    os.makedirs(out, exist_ok=True)
    sel.to_csv(f"{out}/selection.csv", index=False)
    preds.to_csv(f"{out}/predictions.csv", index=False)

    # familiar: mean single-seed MAE per fold (canonical), next to the earlier methods
    fam = preds[(preds["regime"] == "familiar") & (preds["inference"] == "canonical")]
    fold = fam.assign(ae=(fam["y_pred"] - fam["y_true"]).abs()).groupby(["variant", "task", "seed"])["ae"].mean() \
        .groupby(["variant", "task"]).mean().unstack("variant")
    c5 = pd.read_csv(f"{FT}/c5_folds.csv").set_index("task")
    hc = pd.read_csv(f"{FT}/head_control/folds.csv").set_index("task")
    fold["finetuned"] = c5["mean"]
    fold["frozen_best"] = c5["frozen_mae"]
    for m in ("mlp_hidden_1_R1", "mlp_hidden_1_R2", "ridge_hidden_1_matched"):
        fold[m] = hc[m]
    fold.to_csv(f"{out}/familiar_folds.csv")
    familiar = [t for t in all_tasks if t["regime"] == "familiar"]
    n_train = float(np.mean([len(t["fit"]) for t in familiar]))
    n_test = float(np.mean([len(t["test"]) for t in familiar]))
    con = corrected_contrasts(fold.dropna(), n_train, n_test)
    con.to_csv(f"{out}/contrasts.csv", index=False)

    # all regimes, both inference modes: MAE per task (mean single-seed)
    per_task = preds.assign(ae=(preds["y_pred"] - preds["y_true"]).abs()) \
        .groupby(["variant", "regime", "inference", "task", "seed"])["ae"].mean() \
        .groupby(["variant", "regime", "inference", "task"]).mean().rename("mae").reset_index()
    per_task.to_csv(f"{out}/task_mae.csv", index=False)

    pd.set_option("display.width", 200)
    print(fold.agg(["mean", "std"]).T.round(2).to_string())
    k = sel[sel["chosen_lr"]]
    print(k.groupby(["variant", "regime"]).agg(lr=("lr", lambda s: s.value_counts().to_dict()),
                                               best_epoch=("best_epoch", "median"), epochs=("epochs_run", "median"),
                                               at_limit=("epochs_run", lambda s: int((s == 30).sum())),
                                               runs=("epochs_run", "size")).to_string())
    print(per_task.groupby(["variant", "regime", "inference"])["mae"].mean().round(2).to_string())
    print(con.round(3).to_string(index=False))


def main():
    d = load()
    all_tasks = tasks(d)
    if ARGS.collect:
        collect(d, all_tasks, OUT)
    elif ARGS.check_default:
        check_default(d, all_tasks, ARGS.threads)
    elif ARGS.pilot:
        import tempfile
        t = next(t for t in all_tasks if t["task"] == "r0f0")
        start = time.time()
        one_run(d, t, "E1_2r", 1e-4, 0, tempfile.mkdtemp(), ARGS.threads, max_epochs=1)
        print(f"pilot: 1 epoch + predictions in {time.time() - start:.0f}s")
    else:
        if ARGS.phase is None or ARGS.shard is None:
            raise SystemExit("give --phase and --shard (or --collect / --check-default / --pilot)")
        if ARGS.phase == "familiar":
            jobs = [(t, v) for t in all_tasks if t["regime"] == "familiar" for v in VARIANTS]
        else:
            v = chosen_variant(OUT)
            print(f"transfer variant: {v}", flush=True)
            jobs = [(t, v) for t in all_tasks if t["regime"] in ("holdout", "nist")]
        for t, v in jobs[ARGS.shard::ARGS.n_shards]:
            task_runs(d, t, v, OUT, ARGS.threads)


if __name__ == "__main__":
    main()
