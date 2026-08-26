"""Cluster Validity Indices (CVIs) for fuzzy partitions.

Convention: ``U`` is the (c, n) membership matrix (rows = clusters), matching
``FuzzyCMeansEngine.U_``.  ``X`` is (n, d), ``V`` is (c, d).

Indices (formulas numbered in paper/sections/02-formulation.typ):

    FPC   V_PC(U)  = (1/n) sum_ij u_ij^2                    in [1/c, 1]   higher = crisper
    PE    V_PE(U)  = -(1/n) sum_ij u_ij log_a u_ij          in [0, log_a c] lower = crisper
    MPC            = 1 - c/(c-1) (1 - FPC)                   in [0, 1]     removes c-bias
    XB    V_XB     = (sum_ij u_ij^m ||x_j-v_i||^2) / (n * min_{i!=k} ||v_i-v_k||^2)   lower better
    FS    V_FS     = sum_ij u_ij^m (||x_j-v_i||^2 - ||v_i-vbar||^2)                    more neg. better
    Kwon  V_K      = (sum_ij u_ij^2 ||x_j-v_i||^2 + (1/c) sum_i ||v_i-vbar||^2)
                     / min_{i!=k} ||v_i-v_k||^2                                        lower better
"""
from __future__ import annotations

import numpy as np


def _U(U):
    U = np.asarray(U, dtype=float)
    if U.ndim != 2:
        raise ValueError("U must be 2D (c, n)")
    return U


def _sqdist_xv(X, V):
    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    diff = X[:, None, :] - V[None, :, :]          # (n, c, d)
    return np.einsum("ncd,ncd->nc", diff, diff)   # (n, c)


def _min_center_sepsq(V):
    V = np.asarray(V, dtype=float)
    c = V.shape[0]
    if c < 2:
        return np.nan
    d2 = np.sum((V[:, None, :] - V[None, :, :]) ** 2, axis=-1)
    d2[np.diag_indices(c)] = np.inf
    return float(d2.min())


def fpc(U):
    """Fuzzy Partition Coefficient. Higher => crisper (1 for a hard partition)."""
    U = _U(U)
    return float(np.sum(U**2) / U.shape[1])


def partition_entropy(U, a=np.e):
    """Partition Entropy in log base ``a``. Lower => crisper (0 for a hard partition)."""
    U = _U(U)
    nz = U > 0
    t = np.zeros_like(U)
    t[nz] = U[nz] * (np.log(U[nz]) / np.log(a))
    return float(-np.sum(t) / U.shape[1])


def modified_partition_coefficient(U):
    """MPC = 1 - c/(c-1) (1 - FPC); comparable across different c."""
    U = _U(U)
    c = U.shape[0]
    if c < 2:
        return 0.0
    return float(1.0 - c / (c - 1.0) * (1.0 - fpc(U)))


def xie_beni(X, U, V, m=2.0):
    """Xie-Beni index: fuzzy compactness over minimum prototype separation."""
    U = _U(U)
    d2 = _sqdist_xv(X, V)
    sigma = float(np.sum((U.T**m) * d2))
    return sigma / (U.shape[1] * _min_center_sepsq(V))


def fukuyama_sugeno(X, U, V, m=2.0):
    """Fukuyama-Sugeno index: compactness minus spread of prototypes about the grand mean."""
    U = _U(U)
    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    d2 = _sqdist_xv(X, V)
    vbar = X.mean(axis=0)
    spread = np.sum((V - vbar) ** 2, axis=1)      # (c,)
    um = U.T**m                                   # (n, c)
    return float(np.sum(um * d2) - np.sum(um * spread[None, :]))


def kwon(X, U, V):
    """Kwon (1998) index: Xie-Beni with a prototype-dispersion term that curbs the c -> n bias."""
    U = _U(U)
    X = np.asarray(X, dtype=float)
    V = np.asarray(V, dtype=float)
    d2 = _sqdist_xv(X, V)
    compact = float(np.sum((U.T**2) * d2))
    vbar = X.mean(axis=0)
    penalty = float(np.sum((V - vbar) ** 2) / V.shape[0])
    return (compact + penalty) / _min_center_sepsq(V)


def summary(X, U, V, m=2.0):
    """All indices as a dict."""
    return {
        "FPC": fpc(U),
        "MPC": modified_partition_coefficient(U),
        "PE": partition_entropy(U),
        "XB": xie_beni(X, U, V, m),
        "FS": fukuyama_sugeno(X, U, V, m),
        "Kwon": kwon(X, U, V),
    }
