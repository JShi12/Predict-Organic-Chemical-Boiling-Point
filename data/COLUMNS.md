# Column dictionary

Columns that aren't self-explanatory, for the data files that need them.

## `nist_noise_floor_lookups.csv`

NIST WebBook boiling points for a random sample of measured literature compounds, used to estimate the noise
floor of the labels (`scripts/lookup_noise_floor.py`).

| Column | Meaning |
|---|---|
| `mol_id` | Row id in `results/representation_study/splits.csv` |
| `label` | This project's literature boiling point (K) |
| `inchi` | Standard InChI from RDKit, used as the NIST search key (pins the exact structure) |
| `nist_id` | NIST WebBook compound id; empty if not found |
| `status` | `found`, `no_boiling_point` (on NIST, but no boiling-point data), `not_found`, or `error:<message>` |
| `n_measurements` | Number of NIST normal-boiling-point values |
| `nist_values` | Every NIST value (K), separated by `;` |
| `nist_median` | Median of `nist_values` (K) |
