"""Inferential statistics helpers for the mobile-pick poster charts.

Pure numpy/scipy. Used by make_poster_charts_v3.py.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import stats


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval, returns (p, lo, hi) as fractions."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def newcombe_diff(k1, n1, k2, n2, z: float = 1.96):
    """Newcombe method 10 CI for the difference of two independent proportions
    (p1 - p2). Returns (diff, lo, hi) as fractions."""
    p1, l1, u1 = wilson(k1, n1, z)
    p2, l2, u2 = wilson(k2, n2, z)
    diff = p1 - p2
    lo = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (diff, lo, hi)


def mcnemar(paired):
    """paired: list of (a_success:bool, b_success:bool) on the SAME scenes.
    Returns dict with discordant counts, continuity-corrected chi2, exact p,
    and the paired difference p_a - p_b with its CI (Newcombe paired, method 10)."""
    b = sum(1 for a, bb in paired if a and not bb)      # a wins
    c = sum(1 for a, bb in paired if (not a) and bb)     # b wins
    n = len(paired)
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0
    p_chi = float(stats.chi2.sf(chi2, 1)) if (b + c) > 0 else 1.0
    p_exact = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) > 0 else 1.0
    pa = sum(1 for a, _ in paired if a) / n if n else 0.0
    pb = sum(1 for _, bb in paired if bb) / n if n else 0.0
    return {"n": n, "b": b, "c": c, "chi2": chi2, "p_chi2": p_chi,
            "p_exact": p_exact, "diff": pa - pb, "pa": pa, "pb": pb}


def logistic_fit(x, y, iters: int = 50):
    """Newton-Raphson logistic regression y ~ 1 + x (y in {0,1}).
    Returns (b0, b1, se1). Odds ratio per unit x = exp(b1)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu)
        XtWX = X.T @ (X * W[:, None])
        try:
            step = np.linalg.solve(XtWX + 1e-9 * np.eye(2), X.T @ (y - mu))
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = X @ beta; mu = 1 / (1 + np.exp(-eta)); W = mu * (1 - mu)
    cov = np.linalg.inv(X.T @ (X * W[:, None]) + 1e-9 * np.eye(2))
    se1 = math.sqrt(cov[1, 1])
    return beta[0], beta[1], se1, cov


def logistic_band(b0, b1, cov, xgrid, z: float = 1.96):
    """95% CI band for the fitted logistic curve via the delta method.
    Returns (yhat, lo, hi) on the probability scale over xgrid."""
    xgrid = np.asarray(xgrid, float)
    X = np.column_stack([np.ones_like(xgrid), xgrid])
    eta = X @ np.array([b0, b1])
    var_eta = np.einsum("ij,jk,ik->i", X, cov, X)
    se = np.sqrt(np.clip(var_eta, 0, None))
    lo_eta, hi_eta = eta - z * se, eta + z * se
    sig = lambda t: 1 / (1 + np.exp(-t))
    return sig(eta), sig(lo_eta), sig(hi_eta)
