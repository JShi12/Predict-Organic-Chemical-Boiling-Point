"""Exploratory error analysis for the representation study (Step 7).

Uses saved predictions only, except the spelling probe, which refits each
familiar fold's frozen MoLFormer + Ridge model with its saved layer and alpha
(checked to reproduce the saved predictions) and predicts every spelling.

Methods: best frozen pipelines for curated (log), Morgan and MoLFormer; the
fine-tuned model (canonical SMILES; absolute error averaged over its 3 seeds).
Familiar regime: a molecule's error is averaged over its 3 repeats.

Writes results/representation_study/error_analysis/:
- molecules.csv: per regime x molecule x method: error, prediction, distance
  to training, attributes;
- bins.csv: MAE per attribute bin with 95% group-bootstrap intervals, and the
  difference to curated;
- nist_cells.csv: NIST subgroup x single-measurement;
- disagreements.csv: the 20 familiar molecules where frozen MoLFormer and
  curated disagree most;
- spelling.csv: per familiar test molecule, frozen MoLFormer's prediction
  from the canonical SMILES and from each of 10 randomised spellings.

Needs data/cache/molformer/spellings/ (scripts/embed_spellings.py). No torch.

    python scripts/error_analysis.py
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.base import clone

from boiling_point import error_analysis as ea
from boiling_point import features, study
from boiling_point import representations as rep

BASE = "results/representation_study"
OUT = f"{BASE}/error_analysis"
CACHE = "data/cache/molformer"
METHODS = ["curated", "morgan", "molformer_frozen", "molformer_finetuned"]
FROZEN = {"curated": "curated", "morgan": "morgan", "molformer": "molformer_frozen"}
N_TOP = 20


def molecule_table() -> pd.DataFrame:
    lit = pd.read_csv(f"{BASE}/splits.csv")
    nist = pd.read_csv(f"{BASE}/nist_external.csv")
    nist = nist[~nist["exclude"]]
    pubchem = pd.read_csv("data/compound_property_from_PubChem.csv", usecols=["cmpdname", "polararea", "rotbonds"])
    pubchem = pubchem.drop_duplicates("cmpdname").set_index("cmpdname")
    if not lit["cmpdname"].isin(pubchem.index).all():
        raise ValueError("literature compounds missing from the PubChem properties")
    hard = features.in_hard_region(pubchem.loc[lit["cmpdname"]].reset_index(drop=True))
    a = pd.DataFrame({"key": "literature:" + lit["mol_id"].astype(str), "name": lit["cmpdname"],
                      "smiles": lit["isosmiles"].map(rep.model_smiles), "group": lit["family"],
                      "structure_group": "lit:" + lit["structure_group"].astype(str), "hard_region": hard.to_numpy(),
                      "single_measurement": np.nan})
    b = pd.DataFrame({"key": "nist:" + nist.index.astype(str), "name": nist["cmpdname"].to_numpy(),
                      "smiles": nist["isosmiles"].map(rep.model_smiles).to_numpy(), "group": nist["subgroup"].to_numpy(),
                      "structure_group": ("nist:" + nist.index.astype(str)).to_numpy(), "hard_region": np.nan,
                      "single_measurement": nist["single_measurement"].to_numpy()})
    mols = pd.concat([a, b], ignore_index=True)
    return pd.concat([mols, ea.molecule_attributes(mols["smiles"])], axis=1)


def task_errors() -> pd.DataFrame:
    """One row per regime x task x molecule x method."""
    best = pd.read_csv(f"{BASE}/frozen/best_predictions.csv")
    frozen = best.assign(method=best["representation"].map(FROZEN))
    frozen["abs_err"] = (frozen["y_pred"] - frozen["y_true"]).abs()
    ft = pd.read_csv(f"{BASE}/finetune/predictions.csv")
    ft = ft[(ft["inference"] == "canonical") & ft["regime"].isin(["familiar", "holdout", "nist"])]
    ft = (ft.assign(abs_err=(ft["y_pred"] - ft["y_true"]).abs())
          .groupby(["regime", "task", "key"]).agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"),
                                                  abs_err=("abs_err", "mean"), n_seeds=("seed", "nunique"))
          .reset_index().assign(method="molformer_finetuned"))
    if (ft["n_seeds"] != 3).any():
        raise ValueError("fine-tuned predictions without 3 seeds")
    cols = ["regime", "task", "key", "method", "y_true", "y_pred", "abs_err"]
    long = pd.concat([frozen[cols], ft[cols]], ignore_index=True)
    iv = pd.read_csv(f"{BASE}/uncertainty/intervals.csv.gz", usecols=["regime", "task", "method", "variant", "key",
                                                                      "train_distance"])
    dist = iv[(iv["method"] == "curated_frozen") & (iv["variant"] == "cvplus")][["regime", "task", "key",
                                                                                 "train_distance"]]
    out = long.merge(dist, on=["regime", "task", "key"], how="left", validate="many_to_one")
    if out["train_distance"].isna().any():
        raise ValueError("molecules without a training distance")
    return out


def per_molecule(errors: pd.DataFrame, mols: pd.DataFrame) -> pd.DataFrame:
    m = (errors.groupby(["regime", "key", "method"])
         .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"), abs_err=("abs_err", "mean"),
              train_distance=("train_distance", "mean"), n_tasks=("task", "nunique")).reset_index())
    return m.merge(mols, on="key", how="left", validate="many_to_one")


def wide(m: pd.DataFrame, regime: str) -> pd.DataFrame:
    w = m[m["regime"] == regime].pivot(index="key", columns="method", values="abs_err")[METHODS]
    if w.isna().any().any():
        raise ValueError(f"{regime}: a molecule lacks a method")
    attrs = m[m["regime"] == regime].drop_duplicates("key").set_index("key")
    return w.join(attrs.drop(columns=["method", "y_pred", "abs_err", "n_tasks"]))


def binned(w: pd.DataFrame, regime: str, attribute: str, labels: pd.Series, order: list) -> list:
    rows = []
    for label in order:
        sel = (labels == label).to_numpy()
        if not sel.any():
            continue
        res = ea.group_bootstrap_mae(w.loc[sel, METHODS], w.loc[sel, "structure_group"], reference="curated")
        rows.append(res.assign(regime=regime, attribute=attribute, bin=str(label)))
    return rows


def bins_table(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    w = wide(m, "familiar")
    rows += binned(w, "familiar", "all", pd.Series("all", index=w.index), ["all"])
    fam = ["alcohol/phenol", "alkane", "alkene/alkyne", "amine", "other"]
    rows += binned(w, "familiar", "family", w["group"], fam)
    for attr in ("heavy_atoms", "hbd", "hba"):
        labels = ea.assign_bins(w[attr], attr).set_axis(w.index)
        rows += binned(w, "familiar", attr, labels, [b[0] for b in ea.BINS[attr]])
    for attr in ("hard_region", "small_polyol"):
        rows += binned(w, "familiar", attr, w[attr].astype(bool).map({True: "yes", False: "no"}), ["no", "yes"])
    for regime in ("familiar", "holdout", "nist"):
        w = wide(m, regime)
        q = pd.qcut(w["train_distance"], 5, labels=False)          # quintiles; label = observed min–max
        rng = w["train_distance"].groupby(q).agg(["min", "max"])
        names = {k: f"Q{k + 1}: {r['min']:.2f}–{r['max']:.2f}" for k, r in rng.iterrows()}
        labels, order = q.map(names), [names[k] for k in sorted(names)]
        rows += binned(w, regime, "train_distance", labels, order)
    out = pd.concat(rows, ignore_index=True)
    return out[["regime", "attribute", "bin", "method", "n", "n_groups", "mae", "ci_low", "ci_high", "diff",
                "diff_ci_low", "diff_ci_high"]]


def nist_cells(m: pd.DataFrame) -> pd.DataFrame:
    w = wide(m, "nist")
    rows = []
    for (subgroup, single), part in w.groupby(["group", "single_measurement"]):
        res = ea.group_bootstrap_mae(part[METHODS], part["structure_group"], reference="curated")
        rows.append(res.assign(subgroup=subgroup, single_measurement=bool(single)))
    out = pd.concat(rows, ignore_index=True)
    return out[["subgroup", "single_measurement", "method", "n", "mae", "ci_low", "ci_high", "diff", "diff_ci_low",
                "diff_ci_high"]]


def disagreements(m: pd.DataFrame) -> pd.DataFrame:
    f = m[m["regime"] == "familiar"]
    pred = f.pivot(index="key", columns="method", values="y_pred")[METHODS].add_prefix("pred_")
    err = f.pivot(index="key", columns="method", values="abs_err")[METHODS].add_prefix("err_")
    attrs = f.drop_duplicates("key").set_index("key")[["name", "smiles", "group", "y_true", "heavy_atoms", "hbd",
                                                       "n_oh", "hard_region", "small_polyol", "train_distance"]]
    t = attrs.join(pred).join(err)
    # curated with Ridge too: the best curated pipeline is XGBoost, MoLFormer's is Ridge, and trees can't
    # predict beyond the training targets' range, so this separates regressor from representation
    ridge = pd.read_csv(f"{BASE}/frozen/ridge_predictions.csv").query("regime == 'familiar' and "
                                                                      "representation == 'curated'")
    t["pred_curated_ridge"] = ridge.groupby("key")["y_pred"].mean()
    t["err_curated_ridge"] = (t["pred_curated_ridge"] - t["y_true"]).abs()
    t["molformer_minus_curated_pred"] = t["pred_molformer_frozen"] - t["pred_curated"]
    t["closer"] = np.where(t["err_molformer_frozen"] < t["err_curated"], "MoLFormer", "curated")
    t = t.reindex(t["molformer_minus_curated_pred"].abs().sort_values(ascending=False).index).head(N_TOP)
    return t.reset_index()


def spelling_probe() -> pd.DataFrame:
    lit = pd.read_csv(f"{BASE}/splits.csv")
    keys = "literature:" + lit["mol_id"].astype(str)
    smiles = lit["isosmiles"].map(rep.model_smiles)
    y = lit["boiling_point_kelvin"].to_numpy()
    canonical = np.asarray(rep.load_cache(CACHE, keys, smiles, "canonical", rep.checked_config()), dtype=np.float64)
    spell_mols = pd.read_csv(f"{CACHE}/spellings/molecules.csv")
    if spell_mols["key"].tolist() != keys.tolist() or spell_mols["smiles"].tolist() != smiles.tolist():
        raise ValueError("spelling cache does not match splits.csv; rerun scripts/embed_spellings.py")
    spellings = np.load(f"{CACHE}/spellings/embeddings.npy")
    kept = json.load(open(f"{CACHE}/spellings/layers.json"))["layers"]
    best = pd.read_csv(f"{BASE}/frozen/best_selection.csv")
    if (best.query("regime == 'familiar' and representation == 'molformer'")["regressor"] != "ridge").any():
        raise ValueError("expected Ridge as the best familiar MoLFormer pipeline")
    sel = pd.read_csv(f"{BASE}/frozen/ridge_selection.csv").query("regime == 'familiar' and representation == "
                                                                  "'molformer'").set_index("task")
    saved = pd.read_csv(f"{BASE}/frozen/ridge_predictions.csv").query("regime == 'familiar' and representation == "
                                                                      "'molformer'")
    rows, worst = [], 0.0
    for t in study.familiar_tasks(lit):
        layer, params = int(sel.loc[t["task"], "layer"]), json.loads(sel.loc[t["task"], "params"])
        tr, te = t["train"], t["test"]
        model = clone(study.ridge_pipeline("molformer")).set_params(**params).fit(canonical[layer][tr], y[tr])
        pred = model.predict(canonical[layer][te])
        ref = saved[saved["task"] == t["task"]].set_index("key").loc[keys.iloc[te], "y_pred"].to_numpy()
        worst = max(worst, float(np.abs(pred - ref).max()))
        s = spellings[kept.index(layer)][te].astype(np.float64)                 # (m, 10, 768)
        sp = model.predict(s.reshape(-1, s.shape[-1])).reshape(len(te), -1)
        r = pd.DataFrame({"task": t["task"], "repeat": t["repeat"], "key": keys.iloc[te].to_numpy(), "layer": layer,
                          "y_true": y[te], "pred_canonical": pred})
        rows.append(pd.concat([r, pd.DataFrame(sp, columns=[f"pred_spelling_{j}" for j in range(sp.shape[1])])],
                              axis=1))
    if worst > 1e-6:
        raise ValueError(f"refit does not reproduce the saved predictions (max diff {worst:.2e} K)")
    print(f"spelling probe: refit reproduces saved canonical predictions (max diff {worst:.1e} K)")
    out = pd.concat(rows, ignore_index=True)
    sp = out.filter(like="pred_spelling_")
    out["spelling_mean"] = sp.mean(axis=1)
    out["spelling_sd"] = sp.std(axis=1, ddof=1)
    out["spelling_range"] = sp.max(axis=1) - sp.min(axis=1)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    mols = molecule_table()
    errors = task_errors()
    m = per_molecule(errors, mols)
    m.to_csv(f"{OUT}/molecules.csv", index=False)
    print(m.groupby(["regime", "method"])["abs_err"].agg(["mean", "size"]).round(2).to_string())
    b = bins_table(m)
    b.to_csv(f"{OUT}/bins.csv", index=False)
    nist_cells(m).to_csv(f"{OUT}/nist_cells.csv", index=False)
    disagreements(m).to_csv(f"{OUT}/disagreements.csv", index=False)
    s = spelling_probe()
    s.to_csv(f"{OUT}/spelling.csv", index=False)
    print(f"spelling SD: median {s['spelling_sd'].median():.2f} K; share SD > 1 K {np.mean(s['spelling_sd'] > 1):.3f}")


if __name__ == "__main__":
    main()
