"""Algorithmic benchmark: continuous fuzzy membership vs. rigid / alternative partitions
(Module 2.3).

Comparators
-----------
hard_c_means           : k-means (the m -> 1 limit of FCM) -> crisp Voronoi partition
gustafson_kessel       : FCM with a per-cluster, determinant-normalized norm A_i
                         -> ellipsoidal fuzzy clusters
possibilistic_c_means  : Krishnapuram & Keller (1993) -- drops sum_i u_ij = 1,
                         typicality T_ij in [0, 1]
gaussian_mixture       : full-covariance GMM (the Gerlach et al. 2018 approach) --
                         posterior responsibilities gamma_ik replace u_ij

compare(X, c, m) fits all of them plus FCM and returns a DataFrame contrasting:
  * explained_variance   1 - SS_within / SS_total  on the hard (argmax) labels
  * boundary_rate        fraction of points with max_i u_ij < 0.5
  * mean_max_membership   average top membership (1 for a crisp partition)
  * ari / nmi            agreement of hard labels with the FCM reference
  * n_iter, seconds
"""
from __future__ import annotations

import time

import numpy as np

from .distances import squared_distance
from .engine import FuzzyCMeansEngine

_TINY = np.finfo(float).tiny


# --------------------------------------------------------------------------- utils
def explained_variance(X, labels):
    """1 - SS_within / SS_total for a hard labelling."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    ss_total = float(np.sum((X - X.mean(axis=0)) ** 2))
    ss_within = 0.0
    for g in np.unique(labels):
        Xg = X[labels == g]
        if len(Xg):
            ss_within += float(np.sum((Xg - Xg.mean(axis=0)) ** 2))
    return 1.0 - ss_within / max(ss_total, _TINY)


def boundary_rate(U):
    """Fraction of samples whose top membership is < 0.5 (U is (c, n))."""
    return float(np.mean(np.asarray(U, dtype=float).max(axis=0) < 0.5))


def _ari_nmi(a, b):
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        return (float(adjusted_rand_score(a, b)),
                float(normalized_mutual_info_score(a, b)))
    except Exception:
        return (float("nan"), float("nan"))


def _seed_memberships(X, c, m, random_state):
    """k-means++ seeding of prototypes, then one FCM membership update -> U (c, n)."""
    eng = FuzzyCMeansEngine(n_clusters=c, m=m, init="kmeans++")
    eng._A_cache = None
    V0 = eng._seed_centers(X, np.random.default_rng(random_state))
    return eng._update_U(squared_distance(X, V0)), eng


# ------------------------------------------------------------------- hard c-means
def hard_c_means(X, c, n_init=10, max_iter=300, random_state=0):
    """k-means. Returns dict(labels, centers, U, n_iter) with a one-hot U (c, n)."""
    from sklearn.cluster import KMeans

    X = np.asarray(X, dtype=float)
    km = KMeans(n_clusters=c, n_init=n_init, max_iter=max_iter,
                random_state=random_state).fit(X)
    lab = np.asarray(km.labels_)
    U = np.zeros((c, len(X)))
    U[lab, np.arange(len(X))] = 1.0
    return {"labels": lab, "centers": km.cluster_centers_, "U": U,
            "n_iter": int(km.n_iter_)}


# -------------------------------------------------------------- Gustafson-Kessel
def gustafson_kessel(X, c, m=2.0, max_iter=200, tol=1e-5, reg=1e-6,
                     n_init=1, random_state=0):
    """FCM with per-cluster norm  A_i = det(Sigma_i)^(1/d) Sigma_i^{-1}.

    Sigma_i is the fuzzy covariance of cluster i; the determinant normalization keeps
    det(A_i) = 1 (unit-volume ellipsoids).
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(random_state)
    p = 1.0 / (m - 1.0)
    eye = np.eye(d)

    best = None
    for _ in range(max(1, n_init)):
        U, _ = _seed_memberships(X, c, m, int(rng.integers(1 << 30)))
        V = (U ** m @ X) / np.maximum((U ** m).sum(axis=1, keepdims=True), _TINY)
        hist, converged, it = [], False, 0
        for it in range(1, max_iter + 1):
            um = U ** m                                        # (c, n)
            denom = np.maximum(um.sum(axis=1, keepdims=True), _TINY)
            V = (um @ X) / denom                               # (c, d)
            d2 = np.empty((n, c))
            for i in range(c):
                diff = X - V[i]                                # (n, d)
                Si = np.einsum("n,nd,ne->de", um[i], diff, diff) / denom[i, 0]
                Si = Si + reg * eye
                det = float(np.linalg.det(Si))
                if not np.isfinite(det) or det <= 0:
                    Ai = eye
                else:
                    Ai = (det ** (1.0 / d)) * np.linalg.inv(Si)
                d2[:, i] = np.einsum("nd,de,ne->n", diff, Ai, diff)
            U_new = np.zeros((c, n))
            zero = d2 <= 0.0
            deg = zero.any(axis=1)
            ok = ~deg
            if ok.any():
                inv = d2[ok] ** (-p)
                U_new[:, ok] = (inv / inv.sum(axis=1, keepdims=True)).T
            if deg.any():
                z = zero[deg].astype(float)
                U_new[:, deg] = (z / z.sum(axis=1, keepdims=True)).T
            hist.append(float(np.sum((U_new.T ** m) * d2)))
            shift = float(np.abs(U_new - U).max())
            U = U_new
            if shift < tol:
                converged = True
                break
        obj = hist[-1] if hist else float("inf")
        if best is None or obj < best["objective"]:
            best = {"U": U, "centers": V, "labels": U.argmax(axis=0), "n_iter": it,
                    "converged": converged, "objective_history": np.asarray(hist),
                    "objective": obj}
    assert best is not None
    return best


# ------------------------------------------------------- Gaussian mixture (Gerlach)
def gaussian_mixture(X, c, covariance_type="full", n_init=5, random_state=0):
    """Full-covariance GMM -- the mixture-model route to personality types
    (Gerlach et al., 2018). Returns dict with the posterior responsibility matrix
    ``U`` (c, n) = gamma_ik.T, so boundary_rate(U) is  1 - mean(max posterior).
    """
    from sklearn.mixture import GaussianMixture

    X = np.asarray(X, dtype=float)
    gm = GaussianMixture(n_components=c, covariance_type=covariance_type,
                         n_init=n_init, random_state=random_state, reg_covar=1e-6).fit(X)
    gamma = gm.predict_proba(X)                       # (n, c)
    return {"U": gamma.T, "centers": gm.means_, "labels": gamma.argmax(axis=1),
            "n_iter": int(gm.n_iter_), "converged": bool(gm.converged_),
            "bic": float(gm.bic(X)), "aic": float(gm.aic(X))}


# ------------------------------------------------------- possibilistic c-means
def possibilistic_c_means(X, c, m=2.0, max_iter=200, tol=1e-5, K=1.0,
                          n_init=1, random_state=0):
    """Krishnapuram & Keller (1993) PCM.

    T_ij = 1 / (1 + (d_ij^2 / eta_i)^(1/(m-1))),  eta_i from a prior FCM run:
        eta_i = K * sum_j u_ij^m d_ij^2 / sum_j u_ij^m .
    The partition constraint sum_i T_ij = 1 is NOT enforced.
    """
    X = np.asarray(X, dtype=float)
    p = 1.0 / (m - 1.0)

    fcm = FuzzyCMeansEngine(n_clusters=c, m=m, n_init=n_init,
                            random_state=random_state).fit(X)
    U, V = fcm.U_.copy(), fcm.centers_.copy()
    d2 = squared_distance(X, V)
    um = U ** m
    eta = K * (um * d2.T).sum(axis=1) / np.maximum(um.sum(axis=1), _TINY)   # (c,)

    T = 1.0 / (1.0 + (d2 / np.maximum(eta[None, :], _TINY)) ** p)          # (n, c)
    hist, converged, it = [], False, 0
    for it in range(1, max_iter + 1):
        d2 = squared_distance(X, V)
        T = 1.0 / (1.0 + (d2 / np.maximum(eta[None, :], _TINY)) ** p)
        Tm = (T.T) ** m                                                    # (c, n)
        denom = np.maximum(Tm.sum(axis=1, keepdims=True), _TINY)
        V_new = (Tm @ X) / denom
        hist.append(float(np.sum(Tm.T * d2)))
        shift = float(np.abs(V_new - V).max())
        V = V_new
        if shift < tol:
            converged = True
            break
    return {"U": T.T, "centers": V, "labels": T.argmax(axis=1), "n_iter": it,
            "converged": converged, "eta": eta, "objective_history": np.asarray(hist)}


# ----------------------------------------------------------------------- compare
def compare(X, c, m=2.0, random_state=0):
    """Fit FCM + the three comparators at fixed (c, m); return a comparison DataFrame."""
    import pandas as pd

    X = np.asarray(X, dtype=float)

    t = time.perf_counter()
    fcm = FuzzyCMeansEngine(n_clusters=c, m=m, n_init=5, random_state=random_state).fit(X)
    t_fcm = time.perf_counter() - t
    ref = fcm.U_.argmax(axis=0)

    def row(name, U, labels, n_iter, secs):
        ari, nmi = _ari_nmi(np.asarray(labels), ref)
        return {
            "method": name,
            "explained_variance": explained_variance(X, labels),
            "boundary_rate": boundary_rate(U),
            "mean_max_membership": float(np.mean(np.asarray(U).max(axis=0))),
            "ari_vs_fcm": ari, "nmi_vs_fcm": nmi,
            "n_iter": int(n_iter), "seconds": round(secs, 3),
        }

    rows = [row("FCM", fcm.U_, ref, fcm.n_iter_, t_fcm)]

    t = time.perf_counter()
    hcm = hard_c_means(X, c, random_state=random_state)
    rows.append(row("Hard c-means", hcm["U"], hcm["labels"], hcm["n_iter"],
                    time.perf_counter() - t))

    t = time.perf_counter()
    gk = gustafson_kessel(X, c, m=m, n_init=2, random_state=random_state)
    rows.append(row("Gustafson-Kessel", gk["U"], gk["labels"], gk["n_iter"],
                    time.perf_counter() - t))

    t = time.perf_counter()
    pcm = possibilistic_c_means(X, c, m=m, random_state=random_state)
    rows.append(row("PCM", pcm["U"], pcm["labels"], pcm["n_iter"],
                    time.perf_counter() - t))

    t = time.perf_counter()
    gm = gaussian_mixture(X, c, random_state=random_state)
    rows.append(row("GMM (Gerlach)", gm["U"], gm["labels"], gm["n_iter"],
                    time.perf_counter() - t))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from .data import make_synthetic_ipip
    from .preprocess import factor_scores

    Xi, _, _ = make_synthetic_ipip(n=2000, c=5, seed=0)
    Z = factor_scores(Xi)
    print(compare(Z, c=5, m=2.0, random_state=0).round(4).to_string(index=False))
