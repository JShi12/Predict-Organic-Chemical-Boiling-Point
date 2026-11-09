# Column dictionary: representation-study splits

Written by `scripts/build_representation_splits.py`; settings and seeds are in `splits_config.json`.

## `splits.csv` (one row per measured literature compound)

| Column | Meaning |
|---|---|
| `mol_id` | Row id, in the order of `load_study_data(audited=True)` |
| `stereo_free_smiles` | Canonical SMILES without stereochemistry (RDKit `isomericSmiles=False`) |
| `inchikey14` | First block of the InChIKey: connectivity, ignoring stereochemistry and isotopes |
| `structure_group` | Integer id shared by rows with the same `inchikey14`; groups are never split across folds |
| `fold_r0` … `fold_r2` | Test fold (0–4) of the row in each repeat of the familiar-chemistry CV |
| `holdout_family` | Family the row is held out with in leave-one-family-out; empty = left out of that regime |
| `lc_entry_f{k}_d{d}` | Learning curves, repeat-0 fold `k`, random draw `d`: the row is in the training subset of size `n` if this value is ≤ `n`. `-1` = in fold `k`'s test set; the training-fold size = in the full training fold only |

## `nist_external.csv` (external test set; every row kept)

| Column | Meaning |
|---|---|
| `nist_round` | Which NIST collection round found it: `heuristic` or `feasibility-aware` |
| `n_measurements` | Number of NIST values behind the label (the label is their median) |
| `subgroup` | `ring-containing` (any ring), else `halogenated acyclic` (any F/Cl/Br/I), else `other` |
| `exclude`, `exclude_reason` | Left out of the test set, and why (structure in training, duplicate, audit check) |
| `descriptors_ok` | Whether every curated RDKit descriptor is finite for this molecule |
| `single_measurement` | `n_measurements == 1` |
| `nn_tanimoto_train` | Highest Tanimoto similarity (Morgan radius 2, 2048 bits) to any training compound |

## `grouped_vs_ungrouped.csv`

Fold metrics from `scripts/compare_grouped_splits.py`; `splits` is `grouped` (structure-grouped folds) or
`ungrouped` (the feature study's folds). Both use the same 1,250 labels.

## `probes_invariance.csv` (label-free; one row per MoLFormer layer)

| Column | Meaning |
|---|---|
| `layer` | `hidden_0` = token embeddings, `hidden_1`…`hidden_12` = encoder layers, `pooler` = the model's pooled output |
| `retrieval_share` | Share of randomised-SMILES embeddings whose nearest canonical embedding (cosine) is their own molecule's |
| `median_distance_ratio` | Median of (cosine distance canonical ↔ randomised spelling) / (distance to the nearest other molecule) |

## `probes_chain_length.csv` (label-free; one row per homologous series × representation)

| Column | Meaning |
|---|---|
| `spearman_distance_vs_length` | Spearman correlation between chain length and cosine distance from the series' shortest member |
| `median_step_distance_upto_C8` / `_beyond_C8` | Median cosine distance between consecutive homologues, up to / beyond 8 carbons |
| `share_steps_beyond_C8_unchanged` | Share of consecutive pairs beyond C8 with identical vectors (distance < 1e-9) |

## `frozen/` (Step 3: frozen-representation comparison, `scripts/run_representation_study.py`)

Files are prefixed `ridge_` (Stage A), `xgboost_` (Stage B) or `best_` (per task and representation, the
regressor with the lower inner-CV MAE).

| File / column | Meaning |
|---|---|
| `*_predictions.csv` | One row per held-out molecule × task × representation × regressor |
| `regime`, `task` | `familiar` (`task` = `r{repeat}f{fold}`), `holdout` (`task` = held-out family), or `nist` |
| `key` | `literature:<mol_id>` (row of `splits.csv`) or `nist:<row>` (row of `nist_external.csv`) |
| `*_selection.csv` | Chosen configuration per task: `params` (JSON), MoLFormer `layer` (0–13, see `representations.LAYER_NAMES`), best inner-CV MAE, wall time |
| `*_cvplus_test.csv.gz` | For CV+ intervals (Step 5): the chosen pipeline refitted on each inner fold, and its predictions for the task's test molecules |
| `*_cvplus_residuals.csv.gz` | The same inner-fold models' out-of-fold residuals (`y_true − y_pred`) for the training molecules |
| `layer_probe.csv` | Familiar regime, MoLFormer + Ridge: every layer's inner-CV and test MAE at its own inner-CV-best alpha (diagnostic only) |
| `references.csv` | Training mean, molecular-weight-only Ridge, and (familiar only) the feature study's curated (log) + family ensemble |
| `confirmatory_provisional.csv` | C1–C3: `estimate` = MAE(MoLFormer) − MAE(curated), best pipelines (positive = MoLFormer worse); raw p-values, before Holm |

## `finetune/` (Step 4: fine-tuned MoLFormer-XL, `scripts/run_finetuning.py`)

Tasks: the 15 familiar folds, the 4 held-out families, `nist`, and `n250_f{k}` (`regime` = `learning_curve`: the
n = 250 subset of repeat-0 fold k, draw 0, tested on that fold's test molecules). Each task's training rows are
split once (seed 0) into fitting, stopping (~7.5%) and calibration (~7.5%) rows, all whole structure groups.

| File / column | Meaning |
|---|---|
| `runs/{task}__lr{lr}__s{seed}.json` | One training run (a `/` in the task name becomes `-`): per-epoch log, predictions, sizes. `--collect` builds the tables below from these |
| `selection.csv` | One row per run. `chosen_lr`: the task's learning rate (seed 0's lowest `stop_mae_best3`; seeds 1–2 only ran at it). `stop_mae_best3`: mean of the 3 lowest per-epoch stopping-half MAEs. `best_epoch` (0-based), `epochs_run`, `n_fit`/`n_stop`/`n_calibrate`. `seconds` is wall-clock time and includes any time the computer slept |
| `predictions.csv` | Test predictions of the chosen-rate runs, per seed. `inference`: `canonical` (canonical SMILES) or `tta` (mean over 10 randomised spellings, seed 0, the same spellings as the frozen `random10` embeddings). `key` as in `frozen/` |
| `calibration.csv` | Canonical predictions for each task's calibration rows (Step 5's split-conformal intervals), per seed |
| `training_log.csv` | Per epoch: mean training loss (MSE on standardised targets) and stopping-half MAE (K) |
| `frozen_matched_predictions.csv` | Familiar folds: frozen MoLFormer + Ridge refitted on the same fitting rows (layer and alpha re-chosen in the inner CV); shows what the 15% hold-out costs |
| `c5_folds.csv` | Per familiar fold: fine-tuned single-seed MAE, `mean`/`std`/`size` over seeds, and `frozen_mae` (frozen best MoLFormer pipeline, full training fold) |
| `c5_provisional.csv` | C5: `estimate` = fine-tuned − frozen MAE (positive = fine-tuned worse), corrected resampled t-test; raw p, before Holm |

## `uncertainty/` (Step 5: 90% prediction intervals, `scripts/run_uncertainty.py`)

`method`: `curated_frozen`, `morgan_frozen`, `molformer_frozen` (each task's best frozen pipeline, `regressor`
ridge or xgboost), `molformer_finetuned_ensemble` (mean of the 3 seeds, canonical SMILES) or
`molformer_finetuned_seed0`. `variant`: `cvplus` (K-fold CV+ from the 3 inner-fold models), `cvplus_mondrian`
(familiar only: CV+ with only the test molecule's family's training residuals; a family with too few, i.e. "other",
keeps the pooled interval), `split` (constant-width split conformal on the calibration half) or `split_normalised`
(width scaled by the seed SD + its calibration median). The n = 250 tasks are not included.

| File / column | Meaning |
|---|---|
| `intervals.csv.gz` | One row per test molecule × task × method × variant |
| `group` | Family (literature molecules) or NIST subgroup |
| `y_pred` | Point prediction: the refitted frozen pipeline, or the fine-tuned ensemble mean / seed 0 |
| `lower`, `upper` | 90% interval (K) |
| `uncertainty` | SD of the 3 inner-fold models' (frozen) or 3 seeds' (fine-tuned) predictions; empty for seed 0 |
| `train_distance` | 1 − highest count-Tanimoto similarity (Morgan counts, radius 2) to any training molecule of the task |
| `coverage.csv` | Per regime × task × method × variant × `group` (`all` or one family/subgroup): `n`, `coverage` (share inside the interval), `width` (mean, K) |
| `spearman.csv` | Spearman correlation of a `score` (`model` = `uncertainty`, `distance` = `train_distance`) with the absolute error, per task × method; `scope` = `all` or a family/subgroup with ≥ 10 molecules |
| `retention.csv` | Familiar regime: MAE of the `fraction` of test molecules with the lowest score; `score` also `oracle` (sorted by the true error) and `random` (one random order) |
