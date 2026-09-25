# Predict Organic Chemical Boiling Point

[![CI](https://github.com/JShi12/Predict-Organic-Chemical-Boiling-Point/actions/workflows/ci.yml/badge.svg)](https://github.com/JShi12/Predict-Organic-Chemical-Boiling-Point/actions/workflows/ci.yml)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

Predicting the normal boiling point of organic compounds from molecular structure. What began as a model comparison grew into a complete workflow:
- predictive models;
- honest evaluation;
- an audit of which training labels are real measurements;
- active learning and Bayesian optimisation to decide which experiments to run next.

On the 1,251 compounds with measured boiling points, the best models reach **~10.7 K mean absolute error (~22 K RMSE)** under nested, family-stratified cross-validation. Choosing experiments with Bayesian optimisation finds **69% of the compounds that meet a boiling-point spec after measuring only 12% of the candidates**.

## Results at a glance

| Part | Question | Key result |
|---|---|---|
| 1. [`Boiling_Point_Predictor.ipynb`](Boiling_Point_Predictor.ipynb) | Which classic model predicts boiling point best? | Five architectures perform comparably, so a Ridge + XGBoost + neural-network ensemble is used. |
| 2. [`Active_Learning.ipynb`](Active_Learning.ipynb) | Which compounds should be measured next? | Model-chosen labels match random picking's accuracy with ~27–47% fewer labels. A feasibility-aware NIST round found 105 new boiling points in 302 lookups, against 3,553 for a hand-written rule. |
| 3. [`Boiling_Point_RDKit.ipynb`](Boiling_Point_RDKit.ipynb) | Do physically motivated descriptors help, measured without split luck? | Nested CV shows the original single-split result (26 K RMSE) was an easy split. Curated RDKit descriptors win by MAE but lose by RMSE, because of ~20 extreme compounds. |
| 4. [`Label_Audit.ipynb`](Label_Audit.ipynb) | Which labels are real measurements? | 327 of 1,588 labels are group-contribution *estimates*, 100–250 K too high for large molecules; 10 more are wrong. On the 1,251 measured labels, curated descriptors win clearly: **~10.7 K MAE**. |
| 5. [`Bayesian_Optimisation.ipynb`](Bayesian_Optimisation.ipynb) | How few experiments find the compounds that meet a spec? | For a 453–473 K window, batch BO finds 69% of in-spec compounds after 12% of the experiments: 5.4× random, better than every alternative in 20/20 seeds. |

**What the project shows:**
- **Check the labels before comparing models.** The biggest improvement came from finding that a fifth of the labels were computed estimates, not from better features or models.
- **One test split can mislead.** The original 26 K RMSE came from an unusually easy split; repeated, stratified CV gives ~32 K for the same kind of model.
- **Acquisition functions must match the goal.** "Probability of meeting a spec" finds 67% more in-spec compounds than qLogEI aimed at a target value.
- **Real experiments need feasibility built in.** Uncertainty alone chose compounds that decompose before boiling; weighting by a learned chance of success raised the hit rate from 0.5% to 35%.

## 1. Baseline models

![Predicted vs actual boiling point on the original test split](results/images/predicted_vs_actual.png)

- **Setup:** Ridge, Random Forest, XGBoost, an MLP and SVR, tuned with `GridSearchCV` on 12 features. The features are PubChem properties (molecular weight, polar area, H-bond counts, rotatable bonds) plus atom and bond counts from the SMILES string.
- **Model choice:** a split-sensitivity check (10 resampled splits) showed that no architecture is reliably best. So the final model is a simple-averaging **ensemble of Ridge, XGBoost and the MLP**, one per inductive bias.
- **Results:** test RMSE 26.2 K (MAE 15.2 K, R² 0.95) on the original 60/20/20 split. Part 3 shows that split was unusually easy.
- **Feature importance:** molecular weight dominates (0.54 of XGBoost's importance), then oxygen count and H-bond donors. That matches the physics: dispersion forces scale with size, and hydrogen bonding adds on top.

## 2. Active learning: which compounds to measure next

![Active learning curves](results/images/active_learning_curves.png)

- **Simulated campaigns** (10 seeds): model-driven selection reaches random picking's 550-label accuracy with ~290 labels (bootstrap XGBoost) or ~360 (GP uncertainty).
- **The models rediscover the domain rule** (high polar area / long chains) without being told it.
- **The GP was the weaker selector.** Its uncertainty mostly measures distance from labelled data (Spearman ρ ≈ 0.8), while bootstrap disagreement tracks where the model is actually wrong.
- **Real NIST rounds:** selecting by uncertainty alone picked large drug- and dye-like molecules that decompose before boiling (4 hits in 811 lookups). Weighting uncertainty by a learned chance of success (classifier AUC 0.93) found **105 new boiling points in 302 lookups**. These were mostly halogenated compounds, which the literature data lacks entirely. Adding half of them cut the error on the other half from ~82 K to ~23 K.

## 3. Rich features and nested cross-validation

![Outer-fold MAE and RMSE for each feature set and model](results/images/feature_study_folds.png)

- **Evaluation:** nested CV (5 folds × 3 repeats) stratified by chemical family, with tuning inside each training fold. Models are selected by MAE, with RMSE reported alongside.
- **Features:** 43 curated RDKit descriptors, grouped by the intermolecular force each one tracks: size, branching (Wiener index, Kier shape), hydrogen bonding, unsaturation, composition. They resolve most isomer ties: 224 groups of identical feature vectors drop to 71, mostly stereoisomers.
- **On all labels (~16.4 K MAE):** curated descriptors beat the original 12 by MAE (−1.0 K with log-transformed size descriptors, 13/15 folds) but lose by RMSE. The whole RMSE gap comes from ~20 extreme compounds.
- **Design checks:** one global model beats per-family models, and adding the family as an input adds nothing. The data holds only four acyclic families of C/H/N/O compounds.

## 4. Label audit: which boiling points are real measurements?

| Check (no model predictions used as evidence) | Excluded |
|---|---|
| Label identical to its Joback group-contribution estimate to 0.01 K (fewer than one expected by chance) | 327 |
| Hydrocarbon > 80 K below the n-alkane with the same carbon count (99% sit within 40 K) | 7 |
| Contradicted by NIST, among compounds every model mispredicts in the same direction | 3 (2 overlap with the row above) |
| Data-entry errors found in part 1 | 2 |

- **Joback estimates are decent for small molecules but 100–250 K too high beyond ~20 heavy atoms.** Those were exactly the "extreme compounds" the models kept underpredicting.
- **On the 1,251 measured labels, curated descriptors win clearly:** ensemble MAE 10.8 vs 12.9 K, better in 15/15 folds.
- **On the same compounds, training without the suspect labels lowers MAE from 14.0 to 10.8 K,** even with ~20% less training data. The audit outcome for every flagged compound is in [`data/label_audit.csv`](data/label_audit.csv).

## 5. Bayesian optimisation: finding compounds that meet a spec

![BO campaign: in-spec compounds found vs experiments](results/images/bo_spec_window.png)

**The problem:** find compounds with a boiling point of **453–473 K** among the 1,251 measured ones (151 are in spec, at molecular weights 62–354). The campaign runs 10 random starting experiments, then batches of 5, up to 150; the model is a BoTorch GP on the curated descriptors.

| After 150 experiments (20 seeds) | In-spec found (of 151) | vs random |
|---|---|---|
| **BO: probability of meeting the spec** | **103.6 ± 2.6** | **5.4×** |
| Greedy GP (same model, no exploration) | 79.6 ± 6.3 | 4.2× |
| BO: BoTorch qLogEI toward 463 K | 61.9 ± 10.9 | 3.2× |
| Molecular-weight rule | 47.6 ± 8.2 | 2.5× |
| Random | 19.1 ± 4.1 | 1.0× |

- **Exploration is worth ~24 compounds** over the greedy GP.
- **qLogEI stalls** because it chases ever-closer matches to 463 K rather than everything in the window.
- **Blind spot:** small diols (ethylene glycol, propylene glycol) boil like molecules twice their size, and BO rarely finds them.

## Data

- **Literature boiling points:** [`data/compound_boiling_points_from_literature.xlsx`](data/compound_boiling_points_from_literature.xlsx), from a [J. Chem. Educ. article](https://pubs.acs.org/doi/10.1021/acs.jchemed.3c01040). 1,748 records, 1,588 after merging with PubChem on compound name. Part 4 shows 327 of them are estimates.
- **PubChem properties:** the full 77 MB download isn't committed. [`data/pubchem_subset.csv`](data/pubchem_subset.csv) (1.1 MB) holds every row the project uses, and the loader falls back to it automatically, with identical results. `scripts/build_pubchem_subset.py` rebuilds it from the full file.
- **NIST WebBook lookups:** `data/nist_boiling_points_*.csv` and `data/compound_boiling_points_from_nist.csv` hold the heuristic, uncertainty-only and feasibility-aware collection rounds. The scraper respects NIST's 5-second crawl delay and is not affiliated with NIST.

## Setup & usage

```bash
git clone https://github.com/JShi12/Predict-Organic-Chemical-Boiling-Point.git
cd Predict-Organic-Chemical-Boiling-Point
python3.11 -m venv venv && source venv/bin/activate    # Python 3.11+ (BoTorch needs >= 3.10)
pip install -r requirements.txt                        # pinned versions
pytest tests/                                          # 78 tests
jupyter notebook                                       # open any of the five notebooks
```

The notebooks load precomputed results. To regenerate them:

| Script | Produces | Time (9 cores) |
|---|---|---|
| `scripts/run_active_learning_simulation.py` | part 2 simulation | ~10 min |
| `scripts/run_model_driven_nist_round.py` | part 2 NIST rounds (network, resumable) | hours |
| `scripts/run_feature_study.py [--audited]` | parts 3–4 nested CV | ~4 min each |
| `scripts/audit_labels.py` | `data/label_audit.csv` | seconds |
| `scripts/run_bo_campaign.py` | part 5 campaigns | ~3 min |

## Project structure

```
├── *.ipynb                     # the five notebooks (see Results at a glance)
├── src/boiling_point/          # reusable code
│   ├── data.py, features.py, preprocessing.py, models.py, ensemble.py, evaluation.py   # part 1
│   ├── active_learning.py, nist_scraper.py                                             # part 2
│   ├── descriptors.py, families.py, validation.py                                      # part 3
│   ├── audit.py                                                                         # part 4
│   ├── bayes_opt.py                                                                     # part 5
│   └── viz.py                                                                           # shared plots
├── scripts/                    # long-running experiments (table above)
├── tests/                      # unit tests, run in CI
├── data/                       # literature labels, NIST rounds, PubChem subset, label audit
└── results/                    # precomputed results (CSV) and figures (images/)
```

## Next steps

- **Broader chemistry:** rings, aromatics, carbonyls and halogens (the 105 halogenated NIST compounds are a start), where richer descriptors should matter most.
- **Multi-objective and constrained BO:** e.g. a boiling-point window plus a viscosity or safety constraint (qNEHVI), and cost-aware acquisition.
- **Exploration for blind spots:** reserve part of each batch for the most uncertain candidates, to reach separate regions of good candidates such as the small diols.

## License

This project is licensed under the [MIT License](LICENSE).
