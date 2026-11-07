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
