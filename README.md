# Predict Organic Chemical Boiling Point

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

Predicting the boiling point of organic chemical compounds from their molecular structure and physicochemical properties, using classic ML regression models.

![Predicted vs actual boiling point on the test set](results/images/predicted_vs_actual.png)

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Data](#data)
- [Setup & Usage](#setup--usage)
- [Methodology](#methodology)
- [Results](#results)
- [Feature Importance](#feature-importance)
- [Active Learning: Which Compounds to Measure Next](#active-learning-which-compounds-to-measure-next)
- [Conclusions & Future Work](#conclusions--future-work)
- [License](#license)

## Introduction

This project goes through a full ML project cycle: data collection, data pre-processing and feature engineering, model training and validation, and model evaluation. Five classic ML architectures — Linear Regression (Ridge), Random Forest, XGBoost, Neural Network, and Support Vector Regression — were trained and evaluated for predicting the boiling point of organic chemical compounds.

**Summary of results:** since all five architectures showed comparable validation performance (confirmed with a split-sensitivity analysis — repeating the comparison across many resampled splits), a simple-averaging **ensemble of Ridge, XGBoost, and a Neural Network** is used instead of a single "champion" model. The ensemble model achieved an RMSE of ≈26 K, MAE of ≈15 K, and R² of 0.95 on the held-out test set. 

## Project Structure

```
.
├── Boiling_Point_Predicter.ipynb   # main analysis notebook (narrative + EDA)
├── Active_Learning.ipynb           # which compounds to measure next: GP-driven active learning
├── src/boiling_point/              # reusable pipeline code
│   ├── data.py                     # loading & merging datasets
│   ├── features.py                 # SMILES feature engineering, feature/target selection
│   ├── preprocessing.py            # train/val/test split, scaling
│   ├── models.py                   # GridSearchCV training & evaluation per architecture
│   ├── evaluation.py               # split-sensitivity checks, multi-model evaluation helpers
│   ├── ensemble.py                 # simple-averaging ensemble across model families
│   ├── viz.py                      # shared plotting functions
│   ├── active_learning.py          # GP / bootstrap selectors, simulated labelling campaigns
│   └── nist_scraper.py             # NIST WebBook scraper for extending the dataset
├── scripts/
│   ├── scrape_nist_boiling_points.py       # CLI entry point for the NIST scraper (heuristic rule)
│   ├── run_active_learning_simulation.py   # retrospective active-learning benchmark
│   └── run_model_driven_nist_round.py      # real NIST round with GP-chosen compounds
├── tests/                          # unit tests for src/boiling_point
├── results/                       # simulation results (CSV) and exported plots (images/)
├── compound_boiling_points_from_literature.xlsx
├── compound_property_from_PubChem.csv   # not committed — see Data below
├── requirements.txt
├── LICENSE
└── .github/workflows/ci.yml
```

## Data

1. **PubChem** — `compound_property_from_PubChem.csv` was downloaded from [PubChem](https://pubchem.ncbi.nlm.nih.gov/), a trusted chemical database, and contains 324,628 records. String-type columns were manually dropped from the raw download to reduce file size.
2. **Literature** — `compound_boiling_points_from_literature.xlsx` was sourced from a [published academic journal article](https://pubs.acs.org/doi/10.1021/acs.jchemed.3c01040) and contains 1,748 records with high-quality boiling point measurements.

The two datasets are merged on compound name, giving **1,588 entries** used for modeling.

> **Note:** `compound_property_from_PubChem.csv` (~77MB) is not committed to this repository due to its size — it's listed in `.gitignore`. To reproduce this project, download it from PubChem and select the columns `cmpdname, mw, mf, polararea, hbonddonor, hbondacc, rotbonds, heavycnt, isosmiles, charge`, then place it at the repo root (`boiling_point.data.load_pubchem_data` handles the column selection automatically).

### Extending the Dataset

The error analysis below shows compounds with high polar area and/or rotatable bonds are underrepresented and disproportionately account for the largest prediction errors. `scripts/scrape_nist_boiling_points.py` looks up boiling points for exactly that region on the [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/) — a public, experimentally-measured reference source — for PubChem candidates not already in `compound_boiling_points_from_literature.xlsx`.

```bash
python scripts/scrape_nist_boiling_points.py [output_csv]
```

This is a long-running, network-bound script (NIST's `robots.txt` crawl-delay is 5 seconds, and most candidates need 1-2 requests), so it can take several hours for the full candidate pool. Progress is written to the output CSV incrementally, and reruns automatically resume by skipping compound names already recorded there. It's not affiliated with, maintained by, or endorsed by NIST.

## Setup & Usage

```bash
git clone https://github.com/JShi12/Predict-Organic-Chemical-Boiling-Point.git
cd Predict-Organic-Chemical-Boiling-Point

python3.11 -m venv venv          # Python 3.11+ (BoTorch needs >= 3.10)
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt

# place compound_property_from_PubChem.csv at the repo root (see Data above)

jupyter notebook Boiling_Point_Predicter.ipynb
```

Run the test suite:

```bash
pytest tests/
```

## Methodology

* **Approach**
  1. Split the data into training/validation/test (60/20/20).
  2. Cross-validate and tune hyperparameters on the training set with `GridSearchCV` for each candidate architecture.
  3. Compare the tuned models on the validation set — and check, via repeated resampling, whether any architecture is reliably better rather than just luckier on one split.
  4. Re-train the chosen model architecture on training + validation data, then evaluate it on the held-out test set.

* **Five architectures evaluated:** Linear Regression (Ridge), Random Forest, XGBoost, Neural Network (MLP), Support Vector Regression.

* **Feature engineering:** in addition to PubChem physicochemical properties (molecular weight, polar area, H-bond donor/acceptor counts, rotatable bonds, heavy atom count), simple atom/bond-count features (`C_cnt`, `O_cnt`, `N_cnt`, `side_chain_cnt`, `double_bond_cnt`, `triple_bond_cnt`) were derived from each compound's isomeric SMILES string.

* The dataset contains outliers — a small number of compounds with high molecular weight, polar area, and/or rotatable bond count. These were kept rather than removed, since the model should be able to predict for those real (if rare) cases too.

## Results

| Model | Train RMSE (K) | Validation RMSE (K) |
|---|---|---|
| Linear Regression (Ridge) | 37.1 | 47.9 |
| Random Forest | 39.5 | 52.5 |
| XGBoost | 35.7 | 49.0 |
| Neural Network | 29.5 | 45.3 |
| Support Vector Regression | 38.4 | 51.1 |

All models show higher validation error than training error, indicating some overfitting. The **Neural Network** actually achieves the lowest error on *both* the training and validation sets here — though its train→validation gap (15.7 K) is the largest of the five, making it the least stable of the models tested on this particular split.

A repeated-resampling check (refitting each architecture's tuned hyperparameters across 10 different train/val splits by changing the ramdom_state seed) shows the mean ± std validation RMSE ranges overlap substantially across all five architectures — none of the differences above are large relative to the spread, so picking a single "champion" would largely reflect which rows landed in the validation set on this one split, not a real difference in architecture quality. Instead, a **simple-averaging ensemble of Ridge, XGBoost, and the Neural Network** is used — one representative of each genuinely different inductive bias (linear/regularized, tree/boosting, nonlinear), which is what makes averaging predictions useful rather than just noise. Random Forest is left out as redundant with XGBoost (same tree-based family, and XGBoost had the edge in most single-split comparisons); SVR is left out for being the least stable across splits (highest variance, ±6.6 K vs ±3.3-5.3 K for the others) despite a competitive mean.

The ensemble was fit on the combined training + validation set, then evaluated on the untouched test set:

| Metric | Value |
|---|---|
| 5-fold CV RMSE (train+val) | 35.7 K |
| **Test RMSE** | **26.2 K** |
| Test MAE | 15.2 K |
| Test R² | 0.95 |

![Residual plot on the test set](results/images/residual_plot.png)

The test RMSE being lower than the CV RMSE is the same "easy test set" pattern seen throughout this project — with a dataset of only ~1,588 compounds, results are noticeably sensitive to the random split. Among the largest prediction errors, the affected compounds tend to have unusually large polar area and/or rotatable bond counts — the same outlier region noted during EDA. The notebook explores this dataset-size sensitivity further with a learning curve, an alternative split-ratio experiment, and a targeted data-collection effort (105 additional compounds scraped from the NIST Chemistry WebBook specifically in that outlier region).

## Feature Importance

The XGBoost component of the ensemble has a built-in feature importance (mean decrease in impurity), which highlights molecular weight, oxygen atom count, and hydrogen bond donor count as the strongest predictors of boiling point:

![Feature importance from the XGBoost ensemble member](results/images/feature_importance.png)

| Feature | Importance |
|---|---|
| Molecular weight (`mw`) | 0.539 |
| Oxygen atom count (`O_cnt`) | 0.184 |
| H-bond donor count (`hbonddonor`) | 0.072 |
| Polar area (`polararea`) | 0.071 |
| Side-chain count (`side_chain_cnt`) | 0.038 |
| Rotatable bonds (`rotbonds`) | 0.028 |

This is consistent with chemistry theory: molecular weight drives Van der Waals forces, while polarity and oxygen content drive hydrogen bonding and dipole-dipole attraction — the major intermolecular forces governing boiling point.

## Active Learning: Which Compounds to Measure Next

The dataset extension above chose compounds with a hand-written rule (polar area ≥ 40 or rotatable bonds ≥ 14). [`Active_Learning.ipynb`](Active_Learning.ipynb) tests whether letting a model choose improves the ensemble faster. It uses a retrospective simulation: start from 50 random labels out of the 1,693, reveal more in rounds chosen by each strategy, and retrain the Ridge + XGBoost + NN ensemble at checkpoints. Scoring is on a fixed test set stratified on the hard region, over 10 seeds.

Strategies: **random**, the **heuristic rule**, **GP uncertainty** (BoTorch `SingleTaskGP` with an ARD kernel and conditioned greedy batches), and a **bootstrap XGBoost** ensemble (15 bootstrapped models, picking where they disagree most).

![Active learning curves](results/images/active_learning_curves.png)

| To match random's test RMSE at 550 labels | Labels needed | Saving |
|---|---|---|
| Bootstrap XGBoost | ~290 | ~47% |
| GP uncertainty | ~360 | ~35% |
| Heuristic rule | ~400 | ~27% |

- **Choosing which compounds to label pays off.** Between ~200 and ~550 labels every non-random strategy beats random; at 400 labels the GP and bootstrap selectors beat random on all 10 seeds. Seed-to-seed spread also drops from ±2.7 K to ≤ ±0.7 K.
- **The models rediscover the rule.** Without being told about the hard region, both model-driven selectors spend 50–70% of their labels there (random: 22%). In the hard region the heuristic matches the best model.
- **The bootstrap XGBoost ensemble was the best selector**, slightly ahead of the GP. Uncertainty-only GP sampling first chases chemical extremes (ozone, deuterated methane, tetranitromethane), which leaves it *behind* random at 100 labels. A calibration check suggests why. The GP's uncertainty mostly measures distance from labelled compounds (Spearman ρ ≈ 0.8 with distance) and underestimates how much harder the hard region is (~2.3× vs the real ~3×). The bootstrap spread tracks where the ensemble is actually wrong (ρ ≈ 0.6 with its error by 400 labels, vs 0.5 for the GP).
- **The hard region has a ~50 K error floor.** From ~400 labels on, hard-region RMSE stays at 50–52 K for every strategy, all the way to the full pool. With these 12 features, more of the same kind of data doesn't fix it (see Conclusions).

Reproduce with `python scripts/run_active_learning_simulation.py` (~10 min on 9 cores). `scripts/run_model_driven_nist_round.py` is the real-world counterpart: it ranks 10,776 NIST-likely PubChem candidates by selector uncertainty, with no region filter, and looks them up on NIST in rounds. That round hasn't been run yet.

## Conclusions & Future Work

1. Five classic ML models were evaluated for predicting chemical compound boiling points on a dataset of 1,588 entries. Model performance was found to be sensitive to the train/validation/test split, given the modest dataset size — collecting more data is recommended as a follow-up. Since no single architecture was reliably better than the others, a simple-averaging ensemble of Ridge, XGBoost, and a Neural Network is used instead of a single champion model.
2. Feature selection is naturally embedded in the training process of the XGBoost component of the ensemble; the dominant features are molecular weight, oxygen atom count, H-bond donor count, polar area, side-chain count, and rotatable bond count.
3. To further improve model performance, collecting more data — particularly compounds with large polar area and/or rotatable bond counts — is recommended for re-training.
4. Cross-referencing the recurring large-residual outlier compounds against independent sources surfaced likely data-entry errors of roughly 90–100 K in two of them, both understating boiling point: 2,6-Nonadien-1-ol (369.65 K here vs 469.15 K per PubChem's WHO/FAO JECFA citation) and N-Methyldodecylamine (382.15 K here vs 473.15 K per a commercial chemical supplier site). At least part of this dataset's hardest-to-predict cases may reflect mislabeled training data rather than genuine chemical difficulty — a full audit against primary sources is recommended alongside collecting more data.
5. Active learning shows that choosing which compounds to label reaches the same accuracy with ~27–47% fewer labels than random picking, and that the hand-written hard-region rule captures most of that benefit. It also shows that hard-region error plateaus at ~50 K no matter how many of the existing compounds are labelled. This qualifies point 3: richer molecular descriptors and a label audit are likely to matter more than more compounds described by the same 12 features.

## License

This project is licensed under the [MIT License](LICENSE).
