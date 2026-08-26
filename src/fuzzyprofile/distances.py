"""Squared-distance kernels for Fuzzy C-Means, plus covariance regularization.

The FCM objective uses the A-norm

    d_ij^2 = || x_j - v_i ||_A^2 = (x_j - v_i)^T A (x_j - v_i),

with A a symmetric positive-definite matrix:

    A = I               -> Euclidean
    A = Sigma(X)^{-1}    -> global Mahalanobis
    A_i det-normalized   -> Gustafson-Kessel (per-cluster; built in benchmarks.py)

On IPIP-50 the raw covariance can be ill-conditioned (collinear reverse-keyed items,
constant columns after filtering), so `regularized_inverse` floors the eigenvalues
before inverting and reports the conditioning.
"""
from __future__ import annotations

import warnings

import numpy as np

_TINY = np.finfo(float).tiny


def _as_2d(M, name="array"):
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {M.shape}")
    return M


def singularity_report(Sigma, tol=1e-12):
    """Rank / conditioning summary for a covariance matrix.

    Returns a dict with keys: rank, full_rank, cond, min_eig, max_eig, singular.
    """
    Sigma = _as_2d(Sigma, "Sigma")
    w = np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))
    w_min, w_max = float(w.min()), float(w.max())
    scale = max(w_max, 1.0)
    rank = int(np.sum(w > tol * scale))
    cond = (w_max / w_min) if w_min > 0 else np.inf
    return {
        "rank": rank,
        "full_rank": rank == Sigma.shape[0],
        "cond": cond,
        "min_eig": w_min,
        "max_eig": w_max,
        "singular": (not np.isfinite(cond)) or cond > 1.0 / tol,
    }


def regularized_inverse(Sigma, eps=1e-6, cond_warn=1e12):
    """SPD inverse with eigenvalue flooring at ``eps * lambda_max``.

    A singular or near-singular Sigma still yields a usable metric; a warning is
    emitted when the original condition number exceeds ``cond_warn`` or any
    eigenvalue had to be floored.
    """
    Sigma = _as_2d(Sigma, "Sigma")
    Sigma = 0.5 * (Sigma + Sigma.T)
    w, Q = np.linalg.eigh(Sigma)
    w_max = float(w.max()) if w.size else 1.0
    if w_max <= 0:
        warnings.warn("covariance spectrum non-positive; using identity metric", RuntimeWarning)
        return np.eye(Sigma.shape[0])
    floor = eps * w_max
    n_floored = int(np.sum(w < floor))
    cond = w_max / max(float(w.min()), _TINY)
    if cond > cond_warn or n_floored:
        warnings.warn(
            f"ill-conditioned covariance (cond~{cond:.2e}, {n_floored}/{w.size} eigenvalues "
            f"floored at {floor:.2e}); Mahalanobis metric regularized",
            RuntimeWarning,
        )
    w_reg = np.maximum(w, floor)
    return (Q / w_reg) @ Q.T  # Q diag(1/w_reg) Q^T


def covariance_metric(X, eps=1e-6):
    """A = Sigma(X)^{-1}, regularized. Global Mahalanobis norm for FCM."""
    X = _as_2d(X, "X")
    Sigma = np.atleast_2d(np.cov(X, rowvar=False))
    return regularized_inverse(Sigma, eps=eps)


def squared_distance(X, V, A=None):
    """(n, c) matrix of squared A-distances between rows of X and rows of V.

    Parameters
    ----------
    X : ndarray (n, d)
    V : ndarray (c, d)
    A : ndarray (d, d) or None
        SPD norm-inducing matrix; None => identity (Euclidean).

    Fully vectorized via a single (n, c, d) difference tensor.
    """
    X = _as_2d(X, "X")
    V = _as_2d(V, "V")
    if X.shape[1] != V.shape[1]:
        raise ValueError(f"feature mismatch: X has d={X.shape[1]}, V has d={V.shape[1]}")
    diff = X[:, None, :] - V[None, :, :]  # (n, c, d)
    if A is None:
        return np.einsum("ncd,ncd->nc", diff, diff)
    A = _as_2d(A, "A")
    return np.einsum("ncd,de,nce->nc", diff, A, diff)
