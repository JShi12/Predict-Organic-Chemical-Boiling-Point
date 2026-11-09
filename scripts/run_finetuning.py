"""Fine-tune MoLFormer-XL (representation study, Step 4) and compute C5.

Tasks: the 15 familiar-chemistry folds (C5), and, descriptive only, the 4
held-out families, the NIST set, and the n = 250 learning-curve subsets of
the repeat-0 folds (draw 0). Per task: one fixed validation split (15% of
the training rows, grouped and family-stratified, halved into a stopping
and a calibration half); seed 0 at each of 3 learning rates; the rate with
the lowest mean stopping MAE over its best 3 epochs; seeds 1 and 2 at that
rate.

Training runs in shards, each its own process with torch imported first
(the encoder can't share a process with xgboost; see representations.py).
Each run is saved as soon as it finishes, so a shard can be restarted:
    python scripts/run_finetuning.py --shard 0 --n-shards 3   (and 1, 2)
Then, without torch:
    python scripts/run_finetuning.py --collect
merges the runs, fits the matched frozen check (MoLFormer + Ridge on the
same fitting rows) and computes the provisional C5.
"""
import argparse
import sys

ARGS = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shard", type=int)
    parser.add_argument("--n-shards", type=int, default=3)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="1 task, 1 learning rate, 1 seed, 2 epochs")
    parser.add_argument("--threads", type=int, default=3)
    ARGS = parser.parse_args()
    if not ARGS.collect:
        import torch  # noqa: F401  (before boiling_point: see representations.py)

import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "scripts")
from boiling_point import representations as rep, splits, study, study_stats  # noqa: E402

SPLITS_CSV = "results/representation_study/splits.csv"
NIST_CSV = "results/representation_study/nist_external.csv"
FROZEN_BEST = "results/representation_study/frozen/best_predictions.csv"
CACHE_DIR = "data/cache/molformer"
OUT_DIR = "results/representation_study/finetune"
LC_SIZE = 250


def load():
    lit = pd.read_csv(SPLITS_CSV)
    nist_all = pd.read_csv(NIST_CSV)
    nist = nist_all[~nist_all["exclude"]]
    keys = np.array(list("literature:" + lit["mol_id"].astype(str)) + list("nist:" + nist.index.astype(str)))
    smiles = pd.Series(list(lit["isosmiles"]) + list(nist["isosmiles"])).map(rep.model_smiles).to_numpy()
    families = np.array(list(lit["family"]) + ["(nist)"] * len(nist), dtype=object)
    groups = np.concatenate([lit["structure_group"].to_numpy(), -1 - np.arange(len(nist))])
    y = np.concatenate([lit["boiling_point_kelvin"].to_numpy(), nist["boiling_point_kelvin"].to_numpy()])
    return {"lit": lit, "nist": nist, "keys": keys, "smiles": smiles, "families": families, "groups": groups, "y": y}


def tasks(d) -> list:
    lit = d["lit"]
    out = study.familiar_tasks(lit) + study.holdout_tasks(lit) + [study.nist_task(len(lit), np.arange(len(d["nist"])))]
    for k in range(5):
        entry = lit[f"lc_entry_f{k}_d0"].to_numpy()
        out.append({"regime": "learning_curve", "task": f"n{LC_SIZE}_f{k}", "repeat": 0, "fold": k,
                    "train": np.flatnonzero((entry >= 0) & (entry <= LC_SIZE)),
                    "test": np.flatnonzero(lit["fold_r0"].to_numpy() == k)})
    for t in out:
        t["fit"], t["stop"], t["calibrate"] = rep_split(d, t)
    return out


def rep_split(d, t):
    return splits.validation_split(t["train"], d["families"], d["groups"], 0.15, seed=0)


def run_path(out, task, lr, seed):
    # family names like "alcohol/phenol" contain a slash; the task name inside the file keeps it
    return f"{out}/runs/{task.replace('/', '-')}__lr{lr:g}__s{seed}.json"


def train_shard(d, all_tasks, shard, n_shards, out, smoke, threads):
    from boiling_point import finetune
    os.makedirs(f"{out}/runs", exist_ok=True)
    ordered = [t for t in all_tasks if t["regime"] != "learning_curve"] + \
              [t for t in all_tasks if t["regime"] == "learning_curve"]
    mine = ordered[shard::n_shards]
    lrs = (3e-5,) if smoke else finetune.LEARNING_RATES
    for t in mine:
        def run(lr, seed):
            path = run_path(out, t["task"], lr, seed)
            if os.path.exists(path):
                return json.load(open(path))
            s = d["smiles"]
            r = finetune.train_run(s[t["fit"]], d["y"][t["fit"]], s[t["stop"]], d["y"][t["stop"]], s[t["test"]],
                                   s[t["calibrate"]], lr, seed, max_epochs=2 if smoke else finetune.MAX_EPOCHS,
                                   n_threads=threads)
            record = {"regime": t["regime"], "task": t["task"], "lr": lr, "seed": seed,
                      **{k: r[k] for k in ("log", "best_epoch", "best_stop_mae", "stop_mae_best3", "seconds")},
                      "test_keys": d["keys"][t["test"]].tolist(), "test_canonical": r["test_canonical"].tolist(),
                      "test_tta": None if r["test_tta"] is None else r["test_tta"].tolist(),
                      "calibrate_keys": d["keys"][t["calibrate"]].tolist(), "calibrate": r["calibrate"].tolist(),
                      "n_fit": len(t["fit"]), "n_stop": len(t["stop"]), "n_calibrate": len(t["calibrate"])}
            with open(path + ".tmp", "w") as f:
                json.dump(record, f)
            os.replace(path + ".tmp", path)
            print(f"{t['task']} lr={lr:g} seed={seed}: best epoch {r['best_epoch']}, stop MAE "
                  f"{r['best_stop_mae']:.2f} K, {r['seconds']:.0f}s", flush=True)
            return record

        first = [run(lr, 0) for lr in lrs]
        chosen = min(first, key=lambda r: r["stop_mae_best3"])["lr"]
        for seed in ((1, 2) if not smoke else ()):
            run(chosen, seed)


def collect(d, all_tasks, out):
    runs = [json.load(open(f"{out}/runs/{f}")) for f in sorted(os.listdir(f"{out}/runs")) if f.endswith(".json")]
    y_of = dict(zip(d["keys"], d["y"]))
    sel_rows, pred_rows, cal_rows, log_rows = [], [], [], []
    for task, group in pd.DataFrame(runs).groupby("task"):
        first = group[group["seed"] == 0]
        chosen = first.loc[first["stop_mae_best3"].idxmin(), "lr"]
        for _, r in group.iterrows():
            used = r["lr"] == chosen
            sel_rows.append({"regime": r["regime"], "task": task, "lr": r["lr"], "seed": r["seed"], "chosen_lr": used,
                             "stop_mae_best3": r["stop_mae_best3"], "best_stop_mae": r["best_stop_mae"],
                             "best_epoch": r["best_epoch"], "epochs_run": len(r["log"]), "seconds": r["seconds"],
                             "n_fit": r["n_fit"], "n_stop": r["n_stop"], "n_calibrate": r["n_calibrate"]})
            log_rows += [{"task": task, "lr": r["lr"], "seed": r["seed"], **e} for e in r["log"]]
            if not used:
                continue
            for mode in ("canonical", "tta"):
                if r[f"test_{mode}"] is None:
                    continue
                pred_rows.append(pd.DataFrame({"regime": r["regime"], "task": task, "seed": r["seed"], "lr": r["lr"],
                                               "inference": mode, "key": r["test_keys"],
                                               "y_true": [y_of[k] for k in r["test_keys"]],
                                               "y_pred": r[f"test_{mode}"]}))
            cal_rows.append(pd.DataFrame({"task": task, "seed": r["seed"], "key": r["calibrate_keys"],
                                          "y_true": [y_of[k] for k in r["calibrate_keys"]],
                                          "y_pred": r["calibrate"]}))
    os.makedirs(out, exist_ok=True)
    selection = pd.DataFrame(sel_rows)
    predictions = pd.concat(pred_rows, ignore_index=True)
    selection.to_csv(f"{out}/selection.csv", index=False)
    predictions.to_csv(f"{out}/predictions.csv", index=False)
    pd.concat(cal_rows, ignore_index=True).to_csv(f"{out}/calibration.csv", index=False)
    pd.DataFrame(log_rows).to_csv(f"{out}/training_log.csv", index=False)

    # matched frozen check: MoLFormer + Ridge refitted on the same fitting rows (descriptive)
    done = set(predictions["task"])
    familiar = [t for t in all_tasks if t["regime"] == "familiar" and t["task"] in done]
    X = np.asarray(rep.load_cache(CACHE_DIR, d["keys"], d["smiles"], "canonical", rep.checked_config()),
                   dtype=np.float64)
    matched = []
    for t in familiar:
        r = study.run_task({**t, "train": t["fit"]}, "molformer", "ridge", X, d["y"], d["families"], d["groups"])
        matched.append(pd.DataFrame({"task": t["task"], "key": d["keys"][t["test"]], "y_true": d["y"][t["test"]],
                                     "y_pred": r["pred"], "layer": r["layer"]}))
    if matched:
        pd.concat(matched, ignore_index=True).to_csv(f"{out}/frozen_matched_predictions.csv", index=False)

    # C5: per familiar fold, mean single-seed fine-tuned MAE minus the frozen best pipeline's MAE
    ft = predictions[(predictions["regime"] == "familiar") & (predictions["inference"] == "canonical")]
    ft_fold = ft.assign(ae=(ft["y_pred"] - ft["y_true"]).abs()).groupby(["task", "seed"])["ae"].mean() \
        .groupby("task").agg(["mean", "std", "size"])
    frozen = pd.read_csv(FROZEN_BEST).query("regime == 'familiar' and representation == 'molformer'")
    frozen_fold = frozen.assign(ae=(frozen["y_pred"] - frozen["y_true"]).abs()).groupby("task")["ae"].mean()
    table = ft_fold.join(frozen_fold.rename("frozen_mae"), how="inner")
    table.to_csv(f"{out}/c5_folds.csv")
    if len(table) >= 2:
        n_test = ft.groupby("task").size().mean() / ft_fold["size"].mean()
        n_train = np.mean([len(t["train"]) for t in familiar])
        res = study_stats.corrected_resampled_ttest(table["mean"] - table["frozen_mae"], n_train, n_test)
        pd.DataFrame([{"comparison": "C5", "estimate": res["estimate"], "se": res["se"], "t": res["t"],
                       "df": res["df"], "p_raw": res["p"], "n_folds": len(table),
                       "status": "provisional: Holm and the 1.0 K rule applied once C4 exists"}]).to_csv(
            f"{out}/c5_provisional.csv", index=False)
        print(f"C5 (fine-tuned - frozen, {len(table)} folds): {res['estimate']:+.2f} K, SE {res['se']:.2f}, "
              f"p = {res['p']:.3f}")
    summary = predictions.assign(ae=(predictions["y_pred"] - predictions["y_true"]).abs()) \
        .groupby(["regime", "inference", "task", "seed"])["ae"].mean().groupby(["regime", "inference"]) \
        .agg(["mean", "std"]).round(2)
    print(summary.to_string())


def main():
    out = os.path.join(tempfile.gettempdir(), "finetune_smoke") if ARGS.smoke else OUT_DIR
    d = load()
    all_tasks = tasks(d)
    if ARGS.smoke:
        all_tasks = [t for t in all_tasks if t["task"] == "r0f0"]
    if ARGS.collect:
        collect(d, all_tasks, out)
    else:
        if ARGS.shard is None:
            raise SystemExit("give --shard (or --collect)")
        train_shard(d, all_tasks, ARGS.shard, ARGS.n_shards, out, ARGS.smoke, ARGS.threads)


if __name__ == "__main__":
    main()
