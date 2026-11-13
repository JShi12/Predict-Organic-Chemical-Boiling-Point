import numpy as np
import pytest

from boiling_point import conformal


def test_split_conformal_rank_on_hand_example():
    # n = 9, alpha = 0.1: rank ceil(10 * 0.9) = 9 -> the largest residual
    assert conformal.split_conformal_width(np.arange(1, 10), 0.1) == 9
    # n = 19: rank ceil(20 * 0.9) = 18
    assert conformal.split_conformal_width(np.arange(1, 20), 0.1) == 18
    # n = 8 < 1/alpha - 1: no finite interval
    assert conformal.split_conformal_width(np.arange(1, 9), 0.1) == np.inf


def test_cv_plus_on_hand_example():
    # 2 inner models, 9 residuals; with one test point, lower = 1st smallest of (pred - R),
    # upper = 9th smallest of (pred + R)
    inner = np.array([[10.0], [20.0]])
    fold = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    r = np.arange(1, 10, dtype=float)
    lo, hi = conformal.cv_plus_interval(inner, fold, r, 0.1)
    lows = np.r_[10 - r[:4], 20 - r[4:]]
    highs = np.r_[10 + r[:4], 20 + r[4:]]
    assert lo[0] == np.sort(lows)[0] and hi[0] == np.sort(highs)[8]
    lo, hi = conformal.cv_plus_interval(inner, fold[:8], r[:8], 0.1)
    assert np.isneginf(lo[0]) and np.isposinf(hi[0])


def test_cv_plus_with_identical_models_equals_split_width():
    inner = np.tile(np.array([[5.0, 7.0]]), (3, 1))
    r = np.random.default_rng(0).exponential(size=99)
    fold = np.arange(99) % 3
    lo, hi = conformal.cv_plus_interval(inner, fold, r, 0.1)
    w = conformal.split_conformal_width(r, 0.1)
    assert hi == pytest.approx(np.array([5.0, 7.0]) + w)
    assert np.all(hi - lo >= 2 * w - 1e-9)


def _cv_plus_coverage(rng, n=150, m=60, K=3):
    x = rng.uniform(-2, 2, n + m)
    y = 2 * x + rng.normal(size=n + m)
    xtr, ytr, xte, yte = x[:n], y[:n], x[n:], y[n:]
    fold = np.arange(n) % K
    inner, resid = [], np.empty(n)
    for k in range(K):
        coef = np.polyfit(xtr[fold != k], ytr[fold != k], 1)
        inner.append(np.polyval(coef, xte))
        resid[fold == k] = ytr[fold == k] - np.polyval(coef, xtr[fold == k])
    lo, hi = conformal.cv_plus_interval(np.array(inner), fold, np.abs(resid), 0.1)
    return conformal.coverage(yte, lo, hi)


def test_cv_plus_and_split_cover_about_90_percent_on_exchangeable_data():
    rng = np.random.default_rng(1)
    assert np.mean([_cv_plus_coverage(rng) for _ in range(200)]) == pytest.approx(0.9, abs=0.02)
    cov = []
    for _ in range(300):
        e = rng.normal(size=100)
        w = conformal.split_conformal_width(np.abs(e[:60]), 0.1)
        cov.append(np.mean(np.abs(e[60:]) <= w))
    assert np.mean(cov) == pytest.approx(0.9, abs=0.02)


def test_normalised_reduces_to_constant_with_constant_scale():
    r = np.random.default_rng(2).exponential(size=50)
    widths = conformal.normalised_split_conformal(r, np.full(50, 3.0), np.full(4, 3.0), 0.1)
    assert widths == pytest.approx(np.full(4, conformal.split_conformal_width(r, 0.1)))


def test_count_tanimoto_distance():
    train = np.array([[1, 0, 2], [0, 3, 0]])
    test = np.array([[1, 0, 2], [1, 0, 1], [0, 0, 0]])
    d = conformal.count_tanimoto_distance(test, train)
    # identical -> 0; [1,0,1] vs [1,0,2]: min-sum 2 / max-sum 3; all-zero vs anything: similarity 0 here
    assert d[0] == pytest.approx(0) and d[1] == pytest.approx(1 / 3) and d[2] == pytest.approx(1)


def test_retention_curve():
    u = np.array([3, 1, 2, 4.0])
    e = np.array([30, 10, 20, 40.0])
    assert conformal.retention_curve(u, e, [0.25, 0.5, 1.0]) == pytest.approx([10, 15, 25])
