"""Statistical helpers (pure NumPy/SciPy) shared by analyze.py and the tests."""
from __future__ import annotations

import numpy as np
from scipy import stats


def boot_ci(x, stat=np.mean, B=10000, rng=None, alpha=0.05):
    """Percentile bootstrap CI of a statistic (stat must accept axis=1)."""
    x = np.asarray(x, float)
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = rng.integers(0, len(x), (B, len(x)))
    s = stat(x[idx], axis=1)
    return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def boot_diff_draws(a, b, B, rng):
    """Bootstrap draws of mean(a) - mean(b) (independent resampling of each group)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ia = rng.integers(0, len(a), (B, len(a)))
    ib = rng.integers(0, len(b), (B, len(b)))
    return a[ia].mean(1) - b[ib].mean(1)


def boot_diff_ci(a, b, B, rng, alpha=0.05):
    d = boot_diff_draws(a, b, B, rng)
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def hedges_g(a, b):
    """Hedges' g (bias-corrected standardised mean difference), a minus b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if not np.isfinite(sp) or sp == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)))


def safe_mwu(a, b):
    """Two-sided Mann-Whitney U: (U statistic for a, p-value); (nan, nan) if undefined."""
    try:
        r = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(r.statistic), float(r.pvalue)
    except ValueError:
        return float("nan"), float("nan")


def rank_biserial(a, b):
    """Rank-biserial correlation from the Mann-Whitney U of a (positive: a tends to be larger)."""
    u, _ = safe_mwu(a, b)
    return float(2 * u / (len(a) * len(b)) - 1) if np.isfinite(u) else float("nan")


def safe_kruskal(groups):
    try:
        r = stats.kruskal(*groups)
        return float(r.statistic), float(r.pvalue)
    except ValueError:
        return float("nan"), float("nan")


def welch_anova(groups):
    """Welch's one-way ANOVA for k >= 2 groups with unequal variances.

    Returns (F, df1, df2, p). For k = 2 it equals Welch's t-test (F = t^2).
    """
    g = [np.asarray(x, float) for x in groups]
    k = len(g)
    n = np.array([len(x) for x in g], float)
    if k < 2 or np.any(n < 2):
        return float("nan"), float(k - 1), float("nan"), float("nan")
    m = np.array([x.mean() for x in g])
    v = np.array([x.var(ddof=1) for x in g])
    if np.any(v <= 0):
        return float("nan"), float(k - 1), float("nan"), float("nan")
    w = n / v
    W = w.sum()
    M = (w * m).sum() / W
    a = (w * (m - M) ** 2).sum() / (k - 1)
    h = (((1 - w / W) ** 2) / (n - 1)).sum()
    F = a / (1 + 2 * (k - 2) / (k ** 2 - 1) * h)
    df2 = (k ** 2 - 1) / (3 * h)
    return float(F), float(k - 1), float(df2), float(stats.f.sf(F, k - 1, df2))


def holm(p):
    """Holm-Bonferroni adjusted p-values. NaN inputs are treated as p = 1 (conservative)."""
    p = np.asarray(p, float)
    p = np.where(np.isnan(p), 1.0, p)
    m = len(p)
    order = np.argsort(p)
    out = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out
