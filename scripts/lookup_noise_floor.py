"""Collect NIST measurements for a random sample of measured literature
compounds, to estimate the noise floor of the labels.

The sample (100 compounds, stratified by family, fixed seed) is drawn from
results/representation_study/splits.csv without looking at any model
result. Each compound is searched on NIST WebBook by InChI, and every
normal-boiling-point measurement is kept (not just the median), so both the
spread between NIST measurements and the gap between our label and NIST can
be measured. Written row by row to data/nist_noise_floor_lookups.csv and
resumable. Respects NIST's 5-second crawl delay; not affiliated with NIST.

Usage:
    python scripts/lookup_noise_floor.py
"""
import csv
import os

import numpy as np
import pandas as pd
from rdkit import Chem

from boiling_point import nist_scraper, splits

SPLITS_CSV = "results/representation_study/splits.csv"
OUT_CSV = "data/nist_noise_floor_lookups.csv"
N_SAMPLE, SEED = 100, 0
FIELDS = ["mol_id", "cmpdname", "family", "label", "inchi", "nist_id", "status", "n_measurements",
          "nist_values", "nist_median"]


def draw_sample(df: pd.DataFrame) -> pd.DataFrame:
    """N_SAMPLE compounds, family shares kept (largest-remainder quotas)."""
    rng = np.random.default_rng(SEED)
    quota = splits.proportional_quotas(N_SAMPLE, df["family"].value_counts())
    picks = [rng.choice(df.index[df["family"] == f], size=q, replace=False) for f, q in quota.items() if q]
    return df.loc[np.sort(np.concatenate(picks))]


def main():
    sample = draw_sample(pd.read_csv(SPLITS_CSV))
    done = set(pd.read_csv(OUT_CSV)["mol_id"]) if os.path.exists(OUT_CSV) else set()
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not done:
            writer.writerow(FIELDS)
        for i, r in enumerate(sample.itertuples(), 1):
            if r.mol_id in done:
                continue
            inchi = Chem.MolToInchi(Chem.MolFromSmiles(r.isosmiles))
            nist_id, values, status = nist_scraper.lookup_all_boiling_points_by_inchi(inchi)
            median = round(float(np.median(values)), 2) if values else ""
            writer.writerow([r.mol_id, r.cmpdname, r.family, r.boiling_point_kelvin, inchi, nist_id, status,
                             len(values), ";".join(f"{v:g}" for v in values), median])
            f.flush()
            print(f"{i}/{len(sample)} {r.cmpdname}: {status} {len(values)}", flush=True)


if __name__ == "__main__":
    main()
