"""Statistics for the representation study's confirmatory comparisons.

- corrected_resampled_ttest: paired comparison over repeated-CV folds.
  Folds share most of their training data, so the plain t-test's variance is
  too small; Nadeau & Bengio (2003, "Inference for the Generalization Error",
  Machine Learning 52) inflate it by (1/J + n_test/n_train).
- bootstrap_macro_difference: paired difference in MAE averaged with equal
  weight over strata (held-out families, NIST subgroups), resampling
  molecules within each stratum.
- holm: Holm (1979, "A simple sequentially rejective multiple test
  procedure", Scandinavian Journal of Statistics 6) adjusted p-values.
"""
import numpy as np
import pandas as pd
from scipy import stats


def corrected_resampled_ttest(differences, n_train: float, n_test: float) -> dict:
    """differences: one paired metric difference per fold (e.g. MAE_a - MAE_b).
    Two-sided p-value on J - 1 degrees of freedom."""
    d = np.asarray(differences, dtype=float)
    j = len(d)
    var = d.var(ddof=1)
    se = np.sqrt((1 / j + n_test / n_train) * var)
    t = d.mean() / se if se > 0 else (0.0 if d.mean() == 0 else np.copysign(np.inf, d.mean()))
    p = 2 * stats.t.sf(abs(t), df=j - 1)
    return {"estimate": float(d.mean()), "se": float(se), "t": float(t), "df": j - 1, "p": float(p)}


def bootstrap_macro_difference(abs_err_a, abs_err_b, strata, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Macro-averaged paired MAE difference, mean over strata of
    (MAE_a - MAE_b), with a stratified bootstrap over molecules: a 95%
    percentile interval and a two-sided p = 2 * min(P(d* <= 0), P(d* >= 0))."""
    a, b = np.asarray(abs_err_a, float), np.asarray(abs_err_b, float)
    strata = np.asarray(strata)
    levels = sorted(set(strata))
    diffs = {s: (a - b)[strata == s] for s in levels}
    estimate = float(np.mean([d.mean() for d in diffs.values()]))
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot)
    for s in levels:
        d = diffs[s]
        boot += d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    boot /= len(levels)
    p = min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0)))
    return {"estimate": estimate, "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)), "p": float(p),
            "per_stratum": {s: float(d.mean()) for s, d in diffs.items()}}


def holm(pvalues) -> np.ndarray:
    """Holm-adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def finding(adjusted_p: float, estimate: float, alpha: float = 0.05, min_difference: float = 1.0) -> bool:
    """A difference counts only if it survives Holm and is at least the
    pre-set minimum meaningful difference (1.0 K)."""
    return bool(adjusted_p < alpha and abs(estimate) >= min_difference)


def fold_mae_table(predictions: pd.DataFrame, by=("regime", "task", "representation", "regressor")) -> pd.DataFrame:
    """MAE and RMSE per group of out-of-fold predictions."""
    err = predictions["y_pred"] - predictions["y_true"]
    return (predictions.assign(abs_err=err.abs(), sq_err=err ** 2).groupby(list(by))
            .agg(mae=("abs_err", "mean"), rmse=("sq_err", lambda s: float(np.sqrt(s.mean()))), n=("abs_err", "size"))
            .reset_index())


def confirmatory_table(table: pd.DataFrame, n_boot: int = 10_000, alpha: float = 0.05,
                       min_difference: float = 1.0) -> pd.DataFrame:
    """Holm across the confirmatory comparisons and the finding rule.

    table: one row per comparison with `estimate`, `p_raw` and `test`
    ('corrected t' or 'bootstrap'). A bootstrap p of 0 only says p is below
    the resolution 2 / n_boot, so that bound is used instead (conservative)."""
    out = table.copy()
    bound = 2 / n_boot
    out["p_used"] = np.where(out["test"] == "bootstrap", np.maximum(out["p_raw"], bound), out["p_raw"])
    out["p_holm"] = holm(out["p_used"])
    out["finding"] = [finding(p, e, alpha, min_difference) for p, e in zip(out["p_holm"], out["estimate"])]
    return out
