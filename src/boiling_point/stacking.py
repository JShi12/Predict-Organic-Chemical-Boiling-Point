"""Combining representations (representation study, Step 8, exploratory).

- convex_weights: stacking weights (non-negative, summing to 1) for a set of
  base pipelines, fitted on their out-of-fold training predictions. Squared
  error follows Breiman (1996, "Stacked regressions", Machine Learning 24);
  absolute error (a linear program) is the sensitivity check.
- BlockScaler: standardises each column, then divides every column of a block
  by sqrt(the block's column count), so a 768-column embedding block and a
  43-column descriptor block carry equal total variance in a Ridge penalty.
- oof_matrix: out-of-fold training predictions (y - inner residual) as a
  (training keys x pipelines) table.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from sklearn.base import BaseEstimator, TransformerMixin


def convex_weights(P, y, loss: str = "squared") -> np.ndarray:
    """Weights w >= 0, sum(w) = 1, minimising the loss of P @ w against y.
    P: (n, k) base predictions; y: (n,)."""
    P, y = np.asarray(P, float), np.asarray(y, float)
    n, k = P.shape
    if k == 1:
        return np.ones(1)
    if loss == "squared":
        scale = y.std() or 1.0                      # conditioning only; the optimum is unchanged
        A, b = P / scale, y / scale
        res = minimize(lambda w: 0.5 * np.sum((A @ w - b) ** 2) / n, np.full(k, 1 / k),
                       jac=lambda w: A.T @ (A @ w - b) / n, method="SLSQP", bounds=[(0, 1)] * k,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(k)}],
                       options={"ftol": 1e-14, "maxiter": 1000})
        if not res.success:
            raise RuntimeError(f"stacking weights did not converge: {res.message}")
        w = res.x
    elif loss == "absolute":
        # variables: w (k), e_plus (n), e_minus (n); P w + e_plus - e_minus = y
        c = np.concatenate([np.zeros(k), np.ones(2 * n)])
        A_eq = np.vstack([np.hstack([P, np.eye(n), -np.eye(n)]),
                          np.concatenate([np.ones(k), np.zeros(2 * n)])[None, :]])
        b_eq = np.concatenate([y, [1.0]])
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (k + 2 * n), method="highs")
        if not res.success:
            raise RuntimeError(f"absolute-error weights failed: {res.message}")
        w = res.x[:k]
    else:
        raise ValueError(f"unknown loss {loss!r}")
    w = np.clip(w, 0, None)
    return w / w.sum()


class BlockScaler(BaseEstimator, TransformerMixin):
    """blocks: column counts of consecutive feature blocks (must sum to the
    number of columns). Constant columns become 0 and don't count towards
    their block's size."""

    def __init__(self, blocks=()):
        self.blocks = blocks

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        if sum(self.blocks) != X.shape[1]:
            raise ValueError(f"blocks {self.blocks} don't sum to {X.shape[1]} columns")
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        varying = std > 0
        self.scale_ = np.where(varying, std, 1.0)
        factor = np.empty(X.shape[1])
        start = 0
        for size in self.blocks:
            cols = slice(start, start + size)
            factor[cols] = np.sqrt(max(int(varying[cols].sum()), 1))
            start += size
        self.factor_ = factor
        self.varying_ = varying
        return self

    def transform(self, X):
        Z = (np.asarray(X, float) - self.mean_) / self.scale_ / self.factor_
        Z[:, ~self.varying_] = 0.0
        return Z


def oof_matrix(residuals: pd.DataFrame, y_by_key: pd.Series, pipelines) -> pd.DataFrame:
    """residuals: one task's inner residuals (columns key, pipeline, residual),
    one row per training key and pipeline. Returns out-of-fold predictions
    y - residual, indexed by key, one column per pipeline, keys in the same
    order for every pipeline."""
    wide = residuals.pivot(index="key", columns="pipeline", values="residual")[list(pipelines)]
    if wide.isna().any().any():
        raise ValueError("a training key lacks an out-of-fold prediction for some pipeline")
    return wide.rsub(y_by_key.loc[wide.index].to_numpy(), axis=0)
