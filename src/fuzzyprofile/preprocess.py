"""Feature-space construction and numerical-robustness checks (Module 3).

Two projections of the recoded 50-item IPIP matrix into a low-dimensional space:

  * factor_scores(X)   -- direct projection onto the 5 OCEAN scores (item-block means),
                          optionally z-scored. Interpretable; the primary space.
  * spherical_pca(X)   -- PCA retaining >= var_target of the variance, optionally
                          whitened ("spherical"): components scaled to unit variance so
                          Euclidean FCM in PCA space equals Mahalanobis FCM in the
                          original space.

covariance_diagnostics(X) reports whether a Mahalanobis metric is numerically safe
(condition number, rank, smallest eigenvalue) before distances.covariance_metric is used.
"""
from __future__ import annotations

import numpy as np

from .data import FACTORS, ITEMS_PER_FACTOR, factor_scores  # re-export factor_scores

__all__ = ["factor_scores", "spherical_pca", "covariance_diagnostics", "FACTORS"]

_TINY = np.finfo(float).tiny


def spherical_pca(X, var_target=0.95, whiten=True, center=True):
    """PCA projection retaining at least ``var_target`` of total variance.

    Returns
    -------
    Z : ndarray (n, k)
        Scores on the retained components (whitened to unit variance if ``whiten``).
    info : dict
        {"k", "explained_variance_ratio", "cumulative", "components", "mean",
         "singular_values"}
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0) if center else np.zeros(X.shape[1])
    Xc = X - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = s**2
    ratio = var / var.sum()
    cum = np.cumsum(ratio)
    k = int(np.searchsorted(cum, var_target) + 1)
    k = max(1, min(k, len(s)))
    comp = Vt[:k]                                   # (k, d)
    scores = Xc @ comp.T                            # (n, k)
    if whiten:
        scores = scores / np.maximum(s[:k] / np.sqrt(max(len(X) - 1, 1)), _TINY)
    info = {
        "k": k,
        "explained_variance_ratio": ratio[:k],
        "cumulative": float(cum[k - 1]),
        "components": comp,
        "mean": mu,
        "singular_values": s[:k],
    }
    return scores, info


def covariance_diagnostics(X, tol=1e-12):
    """Conditioning report for cov(X); decides whether global Mahalanobis is safe."""
    X = np.asarray(X, dtype=float)
    Sigma = np.atleast_2d(np.cov(X, rowvar=False))
    w = np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))
    w_min, w_max = float(w.min()), float(w.max())
    scale = max(w_max, 1.0)
    rank = int(np.sum(w > tol * scale))
    cond = (w_max / w_min) if w_min > 0 else np.inf
    return {
        "d": Sigma.shape[0],
        "rank": rank,
        "full_rank": rank == Sigma.shape[0],
        "cond": cond,
        "min_eig": w_min,
        "max_eig": w_max,
        "mahalanobis_safe": np.isfinite(cond) and cond < 1e10 and rank == Sigma.shape[0],
    }


def item_variances(X):
    """Per-item response variance -- flags near-constant items (Mahalanobis hazard)."""
    X = np.asarray(X, dtype=float)
    return X.var(axis=0)


if __name__ == "__main__":
    from .data import make_synthetic_ipip

    Xi, _, _ = make_synthetic_ipip(n=3000, c=5, seed=0)
    S = factor_scores(Xi)
    Z, info = spherical_pca(Xi, var_target=0.95)
    print(f"factor scores: {S.shape}")
    print(f"spherical PCA: {Z.shape}  (k={info['k']}, cum var={info['cumulative']:.3f})")
    print("cov diagnostics (factor scores):", {k: (round(v, 4) if isinstance(v, float) else v)
                                               for k, v in covariance_diagnostics(S).items()})
    print("cov diagnostics (50 items):", {k: (round(v, 2) if isinstance(v, float) else v)
                                          for k, v in covariance_diagnostics(Xi).items()})
