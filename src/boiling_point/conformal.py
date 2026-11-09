"""Conformal prediction intervals for the representation study (Step 5).

- K-fold CV+ (Barber, Candès, Ramdas & Tibshirani 2021, "Predictive inference
  with the jackknife+", Annals of Statistics 49): for frozen models, from the
  K inner-fold models' test predictions and each training molecule's residual
  under the inner model that didn't see it.
- Split conformal (Lei, G'Sell, Rinaldo, Tibshirani & Wasserman 2018,
  "Distribution-free predictive inference for regression", JASA 113): for the
  fine-tuned model, from a held-out calibration set's absolute residuals;
  constant width, or normalised by a per-molecule scale (Papadopoulos,
  Gammerman & Vovk 2008).

An interval that can't be finite at level alpha (too few residuals) is
(-inf, inf).
"""
import math

import numpy as np


def _lower_rank(alpha: float, n: int) -> int:
    """1-based rank of the lower CV+ quantile, floor(alpha (n + 1))."""
    return math.floor(alpha * (n + 1) + 1e-9)


def _upper_rank(alpha: float, n: int) -> int:
    """1-based rank of the upper quantile, ceil((1 - alpha)(n + 1))."""
    return math.ceil((1 - alpha) * (n + 1) - 1e-9)


def cv_plus_interval(inner_test, residual_fold, residual_abs, alpha: float = 0.1):
    """CV+ intervals for m test points.

    inner_test: (K, m) predictions of the K inner-fold models.
    residual_fold: (n,) inner fold index of each training residual (the model
    that did not see that molecule); residual_abs: (n,) its |y - prediction|.
    Returns (lower, upper), each (m,)."""
    inner_test = np.atleast_2d(np.asarray(inner_test, dtype=float))
    fold = np.asarray(residual_fold, dtype=int)
    r = np.abs(np.asarray(residual_abs, dtype=float))
    n, m = len(r), inner_test.shape[1]
    lo_k, hi_k = _lower_rank(alpha, n), _upper_rank(alpha, n)
    if lo_k < 1 or hi_k > n:
        return np.full(m, -np.inf), np.full(m, np.inf)
    centres = inner_test[fold]                              # (n, m)
    lower = np.partition(centres - r[:, None], lo_k - 1, axis=0)[lo_k - 1]
    upper = np.partition(centres + r[:, None], hi_k - 1, axis=0)[hi_k - 1]
    return lower, upper


def split_conformal_width(calibration_abs, alpha: float = 0.1) -> float:
    """Half-width: the ceil((n + 1)(1 - alpha))-th smallest calibration
    |residual|; inf if that rank exceeds n."""
    r = np.sort(np.abs(np.asarray(calibration_abs, dtype=float)))
    k = _upper_rank(alpha, len(r))
    return float(r[k - 1]) if k <= len(r) else np.inf


def normalised_split_conformal(calibration_abs, calibration_scale, test_scale, alpha: float = 0.1,
                               beta: float = None) -> np.ndarray:
    """Per-test-point half-widths q (s + beta), where q is the split-conformal
    quantile of |residual| / (s + beta) on the calibration set. beta defaults
    to the median calibration scale, so a zero scale never gives a zero width."""
    s_cal = np.asarray(calibration_scale, dtype=float)
    beta = float(np.median(s_cal)) if beta is None else beta
    q = split_conformal_width(np.abs(np.asarray(calibration_abs, dtype=float)) / (s_cal + beta), alpha)
    return q * (np.asarray(test_scale, dtype=float) + beta)


def coverage(y, lower, upper) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((y >= lower) & (y <= upper)))


def count_tanimoto_distance(test_fp, train_fp, chunk: int = 8) -> np.ndarray:
    """1 - the highest count-Tanimoto similarity, sum(min) / sum(max), of each
    test fingerprint to any training fingerprint."""
    test_fp, train_fp = np.asarray(test_fp, dtype=np.float32), np.asarray(train_fp, dtype=np.float32)
    best = np.zeros(len(test_fp))
    for i in range(0, len(test_fp), chunk):
        a = test_fp[i:i + chunk, None, :]
        num = np.minimum(a, train_fp[None]).sum(-1)
        den = np.maximum(a, train_fp[None]).sum(-1)
        best[i:i + chunk] = np.max(np.where(den > 0, num / np.where(den > 0, den, 1), 1.0), axis=1)
    return 1 - best


def retention_curve(uncertainty, abs_error, fractions) -> np.ndarray:
    """MAE of the most confident share f of points (lowest uncertainty first),
    for each f in fractions. Ties broken by input order (stable sort)."""
    order = np.argsort(np.asarray(uncertainty, dtype=float), kind="stable")
    e = np.asarray(abs_error, dtype=float)[order]
    return np.array([e[:max(1, int(round(f * len(e))))].mean() for f in fractions])
