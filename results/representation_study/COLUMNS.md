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

## `learning_curve/` (Step 6: learning curves and C4, `scripts/run_learning_curves.py`)

Tasks `n{size}_f{fold}_d{draw}`: repeat-0 fold `fold` trained on its saved nested subset (`lc_entry_f{fold}_d{draw}`
in `splits.csv`) of `size` 100, 250 or 500, tested on the unchanged fold. `size` = `full` rows come from Step 3
(repeat 0, one draw).

| File / column | Meaning |
|---|---|
| `runs/*.json` | One (task, representation, regressor) fit; the tables below are built from these |
| `selection.csv` | Per fit: `n_train` (realised subset size), MoLFormer `layer`, `params`, `inner_mae`, `seconds` |
| `predictions.csv` / `best_predictions.csv` | Test predictions of every fit / of the best pipeline per task and representation (lower inner-CV MAE; `best_selection.csv`) |
| `references.csv` | Training-mean prediction at each task |
| `curves.csv` | MAE per `size` × `fold` × `draw` × `representation` × `regressor` (`best`, `ridge`, `xgboost`, or `reference` for the training mean) |
| `differences.csv` | Per run, best pipelines: MoLFormer − curated and Morgan − curated MAE (K) |
| `c4_provisional.csv` | C4 at n = 100: corrected resampled t-test over the 15 runs (`n_train`, `n_test` used in the correction); `min_detectable` = SE × t(0.975, df) |
| `h5_finetuned_vs_frozen.csv` | Per repeat-0 fold at n = 250 (draw 0) and full size: fine-tuned single-seed MAE (mean over seeds), the frozen best MoLFormer pipeline, and frozen MoLFormer + Ridge refitted on the fine-tuned model's fitting rows (`frozen_matched`); differences `minus_best`, `minus_matched` |

## `confirmatory.csv` (`scripts/confirmatory.py`)

C1–C5 with Holm (1979) at α = 0.05. `estimate`: the first-named method's MAE minus the second's (positive = first
worse). `test`: `corrected t` (Nadeau & Bengio 2003) or `bootstrap` (stratified, 10,000 draws). `p_used`: `p_raw`,
except a bootstrap p of 0 is replaced by its resolution bound 0.0002. `p_holm`: Holm-adjusted. `finding`: `p_holm`
< 0.05 and |`estimate`| ≥ 1.0 K.

## `morgan_descriptive.csv` (`scripts/morgan_descriptive.py`)

Descriptive contrasts for the pre-registered Morgan baseline, outside the confirmatory family (no Holm, no `finding`).
One row per contrast × setting.

| Column | Meaning |
|---|---|
| `contrast` | `Morgan - curated (log)` or `Morgan - frozen MoLFormer`; best pipelines (frozen) |
| `as_in` | the confirmatory comparison whose setting and test are reused (C1–C4) |
| `setting` | familiar 15 folds; held-out families (macro-average of 4); NIST (macro-average of 3 subgroups); n = 100 (15 runs) |
| `test` | `corrected t` (Nadeau & Bengio 2003) or `bootstrap` (stratified, 10,000 draws) |
| `estimate` | MAE(Morgan) − MAE(other), K; positive = Morgan worse |
| `se` / `ci_low`, `ci_high` | SE (corrected t rows) / 95% percentile interval (bootstrap rows) |
| `p_raw` | two-sided p, uncorrected; a bootstrap p of 0 means p < 0.0002 |
| `min_detectable` | n = 100 rows only: SE × t(0.975, df) |
| `n_units` | folds/runs (t) or molecules (bootstrap) |
| `per_stratum` | bootstrap rows: JSON of each family's or subgroup's difference (K) |
| `at_least_1K` | |`estimate`| ≥ 1.0 K |
| `direction` | `Morgan worse` / `Morgan better` |

## `error_analysis/` (Step 7: exploratory error analysis, `scripts/error_analysis.py`)

`method`: `curated`, `morgan`, `molformer_frozen` (best frozen pipelines) or `molformer_finetuned` (canonical SMILES;
`abs_err` = mean over the 3 seeds of the absolute error, `y_pred` = mean over seeds). Intervals: 95% percentile
intervals from 2,000 bootstrap resamples of structure groups (NIST: each molecule its own group); `diff` =
MAE(method) − MAE(curated) on the same draws.

| File / column | Meaning |
|---|---|
| `molecules.csv` | One row per regime × molecule × method. Familiar: averaged over the 3 repeats (`n_tasks`) |
| `abs_err`, `y_pred`, `train_distance` | Absolute error (K), prediction (K), 1 − highest count-Tanimoto similarity to the task's training set |
| `group` | Family (literature) or NIST subgroup; `structure_group` = bootstrap unit |
| `heavy_atoms`, `hbd`, `hba`, `n_oh` | RDKit heavy atoms, Lipinski H-bond donors / acceptors, aliphatic + aromatic OH groups |
| `hard_region` | Literature only: PubChem polar area ≥ 40 or rotatable bonds ≥ 14 (`features.in_hard_region`) |
| `small_polyol` | ≥ 2 OH and ≤ 8 heavy atoms |
| `single_measurement` | NIST only: the label rests on one measurement |
| `bins.csv` | Per `regime` × `attribute` × `bin` × `method`: `n`, `n_groups`, `mae`, `ci_low`, `ci_high`, `diff`, `diff_ci_low`, `diff_ci_high`. `train_distance` bins are quintiles within the regime, labelled with their observed range |
| `nist_cells.csv` | The same statistics per NIST `subgroup` × `single_measurement` |
| `disagreements.csv` | The 20 familiar molecules with the largest \|`pred_molformer_frozen` − `pred_curated`\| (means over repeats): `pred_*`, `err_*` per method, plus `pred_curated_ridge` / `err_curated_ridge` (curated with Ridge, to separate regressor from representation) and `closer` (MoLFormer vs curated best pipeline) |
| `spelling.csv` | Per familiar task × test molecule: `layer` and frozen MoLFormer + Ridge refitted with the saved layer and alpha; `pred_canonical` (equals the saved prediction), `pred_spelling_0`…`9` (the 10 randomised spellings of the cache), `spelling_mean`, `spelling_sd` (ddof = 1), `spelling_range` |

## `combined/` (Step 8: combining representations, exploratory, `scripts/run_combined.py`)

`method`:
- `curated_best`, `molformer_best`, `morgan_best`: the best frozen pipeline per representation, as in `frozen/`.
- `S0`: curated Ridge + curated XGBoost.
- `S1`: best curated + best MoLFormer.
- `S1m`: best curated + best Morgan.
- `S2`: all six frozen pipelines.
- `S1_abs`, `S0_abs`: S1 / S0 with weights that minimise absolute instead of squared error.
- `average`: S1's two inputs with equal weights.
- `concat_ridge`: Ridge on curated (log) + one MoLFormer layer, each block standardised and divided by √(column count).

Stacking weights are non-negative, sum to 1 and are fitted per task on the out-of-fold training predictions
(y − residual from `frozen/*_cvplus_residuals.csv.gz`).

| File / column | Meaning |
|---|---|
| `concat_ridge_predictions.csv`, `concat_ridge_selection.csv` | As `frozen/ridge_predictions.csv` / `ridge_selection.csv`, `representation` = `concat`; `layer` is the MoLFormer layer joined to the descriptors |
| `stacked_predictions.csv` | One row per `regime` × `task` × `method` × test `key`: `y_true`, `y_pred` (K); stacked methods and the three `*_best` |
| `weights.csv` | Per `regime` × `task` × `method` × input `pipeline` (`{representation}_{regressor}`): `weight`; `oof_mae` = the blend's MAE on the out-of-fold training predictions |
| `metrics.csv` | Per `regime` × `method`: `mae` (mean over tasks), `sd` (SD over tasks), `n_tasks`; `nist_macro_mae` = mean of the 3 NIST subgroup MAEs |
| `contrasts.csv` | `contrast` = MAE(first) − MAE(second); `setting_as_in` = the C1–C3 setting whose test is used (`test`: corrected t or stratified bootstrap); `estimate`, `se` (corrected t only), `ci_low`, `ci_high` (familiar: estimate ± t(0.975, 14) × se), `p_raw` (no Holm), `n_units`, `per_stratum`; `at_least_1K_better` = estimate ≤ −1.0 K and `ci_high` < 0 |

## `finetune/head_control/` (exploratory: encoder, head and layer in C5, `scripts/run_head_control.py`)

`method`:
- `mlp_{hidden_1,pooler}_{R1,R2}`: the fine-tuning head (`finetune.make_head`) trained on frozen canonical embeddings
  at layer 1 or the pooled output. R1 is the fine-tuning recipe (rates 1e-5/3e-5/1e-4, ≤ 30 epochs, patience 5).
  R2 gives the head room to converge (1e-4/3e-4/1e-3/3e-3, ≤ 300 epochs, patience 20).
- `ridge_{layer}_matched`: Ridge with the layer probe's alpha, refitted on the same fit rows (~85%).
- `ridge_{layer}_full`: the layer probe's own test MAE (full training fold).
- `finetuned`, `frozen_best`: from `finetune/c5_folds.csv`.

| File / column | Meaning |
|---|---|
| `runs.csv` | One row per head run: `task`, `layer`, `recipe`, `lr`, `seed`, `best_epoch`, `epochs_run`, `best_stop_mae`, `stop_mae_best3` (rate choice), `seconds`, `chosen_lr` |
| `predictions.csv` | Test predictions of the kept runs and matched Ridge: `task`, `method`, `seed`, `key`, `y_true`, `y_pred` (K) |
| `folds.csv` | Per familiar fold (`task`): test MAE (K) per method; MLP = mean single-seed MAE over 3 seeds |
| `contrasts.csv` | `contrast`, `first` − `second` over 15 folds: `estimate`, `se`, `ci_low`/`ci_high` (estimate ± t(0.975, 14) × se), `p_raw` (corrected resampled t-test, no Holm), `folds_first_better`, `n_folds` |
