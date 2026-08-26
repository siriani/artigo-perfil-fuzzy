"""Fuzzy C-Means (Bezdek, 1981) — transparent NumPy implementation.

Objective
---------
    J_m(U, V) = sum_{i=1..c} sum_{j=1..n} u_ij^m * || x_j - v_i ||_A^2

subject to the probabilistic partition constraint  sum_i u_ij = 1  for every j,
u_ij in [0, 1], and 0 < sum_j u_ij < n.

Alternating optimization (Picard iteration) of the two stationarity conditions
obtained via Lagrange multipliers (see paper/sections/02-formulation.typ):

    v_i  = ( sum_j u_ij^m x_j ) / ( sum_j u_ij^m )                         (centers)
    u_ij = 1 / sum_{k=1..c} ( d_ij^2 / d_kj^2 )^{1/(m-1)}                  (memberships)

Stopping rule:  || U^{(k+1)} - U^{(k)} ||_inf < tol   or   k >= max_iter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distances import covariance_metric, squared_distance

_TINY = np.finfo(float).tiny


@dataclass
class _RunResult:
    U: np.ndarray          # (c, n)
    V: np.ndarray          # (c, d)
    n_iter: int
    converged: bool
    history: np.ndarray    # (n_iter,)
    objective: float


class FuzzyCMeansEngine:
    """Fuzzy C-Means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of fuzzy partitions c.
    m : float > 1
        Fuzzifier (weighting exponent). m -> 1+ approaches a hard partition;
        m -> inf drives every membership to 1/c.
    max_iter : int
        Maximum Picard sweeps.
    tol : float
        Convergence threshold on || U^{k+1} - U^k ||_inf.
    metric : {"euclidean", "mahalanobis"}
        Norm used for d_ij. "mahalanobis" uses A = Sigma(X)^{-1} unless `A` is given.
    A : ndarray (d, d) or None
        Explicit SPD norm-inducing matrix (overrides `metric` computation).
    init : {"random", "sample"}
        "random": Dirichlet(1) rows for U^{(0)}.
        "sample": c distinct data points as initial centers, then one U update.
    n_init : int
        Restarts; the run with the lowest final J_m is kept.
    random_state : int or None
        Seed for reproducibility.

    Attributes (after `fit`)
    ------------------------
    U_ : ndarray (c, n)              fuzzy partition matrix
    memberships_ : ndarray (n, c)    U_.T, convenience view
    centers_ : ndarray (c, d)        prototypes V
    n_iter_ : int                    sweeps performed in the retained run
    converged_ : bool                whether the tol criterion was met
    objective_ : float              final J_m
    objective_history_ : ndarray    J_m after each sweep (retained run)
    A_ : ndarray or None            the norm matrix actually used
    """

    def __init__(
        self,
        n_clusters=5,
        m=2.0,
        max_iter=150,
        tol=1e-5,
        tol_obj=1e-7,
        metric="euclidean",
        A=None,
        init="random",
        n_init=1,
        random_state=None,
    ):
        self.tol_obj = float(tol_obj)
        if not m > 1.0:
            raise ValueError("fuzzifier m must be strictly > 1")
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.metric = str(metric)
        self.A = None if A is None else np.asarray(A, dtype=float)
        self.init = str(init)
        self.n_init = int(n_init)
        self.random_state = random_state

    # ------------------------------------------------------------------
    # core updates
    # ------------------------------------------------------------------
    def _resolve_A(self, X):
        if self.A is not None:
            return self.A
        if self.metric == "euclidean":
            return None
        if self.metric == "mahalanobis":
            return covariance_metric(X)
        raise ValueError(f"unknown metric {self.metric!r}")

    def _seed_centers(self, X, rng):
        """Pick c initial prototypes from the data.

        "kmeans++": D^2-weighted seeding (Arthur & Vassilvitskii) — breaks symmetry
        and spreads the seeds, which avoids the trivial FCM fixed point where every
        prototype collapses onto the global mean.
        "random": c distinct data points, uniform.
        """
        n = X.shape[0]
        c = self.n_clusters
        if self.init == "random":
            return X[rng.choice(n, size=c, replace=(n < c))].copy()
        if self.init in ("kmeans++", "sample"):
            first = rng.integers(n)
            centers = [X[first]]
            closest = squared_distance(X, X[first][None, :], self._A_cache)[:, 0]
            for _ in range(1, c):
                total = float(closest.sum())
                probs = np.full(n, 1.0 / n) if total <= 0 else closest / total
                nxt = rng.choice(n, p=probs)
                centers.append(X[nxt])
                d2n = squared_distance(X, X[nxt][None, :], self._A_cache)[:, 0]
                closest = np.minimum(closest, d2n)
            return np.asarray(centers)
        raise ValueError(f"unknown init {self.init!r}")

    def _init_state(self, X, rng):
        """Return an initial (c, n) membership matrix U^{(0)} from seeded prototypes."""
        V0 = self._seed_centers(X, rng)
        return self._update_U(squared_distance(X, V0, self._A_cache))

    def _update_centers(self, X, U):
        um = U**self.m                          # (c, n)
        denom = np.maximum(um.sum(axis=1, keepdims=True), _TINY)
        return (um @ X) / denom                 # (c, d)

    def _update_U(self, d2):
        """Membership update from an (n, c) squared-distance matrix -> (c, n)."""
        n, c = d2.shape
        U = np.empty((c, n))
        zero = d2 <= 0.0
        deg = zero.any(axis=1)                  # rows sitting exactly on >=1 prototype
        ok = ~deg
        if ok.any():
            p = 1.0 / (self.m - 1.0)
            inv = d2[ok] ** (-p)               # (n_ok, c)
            U[:, ok] = (inv / inv.sum(axis=1, keepdims=True)).T
        if deg.any():
            z = zero[deg].astype(float)        # split mass over coincident prototypes
            U[:, deg] = (z / z.sum(axis=1, keepdims=True)).T
        return U

    def _objective(self, U, d2):
        return float(np.sum((U.T**self.m) * d2))

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def _single_run(self, X, A, rng):
        self._A_cache = A
        U = self._init_state(X, rng)
        history = []
        converged = False
        k = 0
        for k in range(1, self.max_iter + 1):
            V = self._update_centers(X, U)
            d2 = squared_distance(X, V, A)
            U_new = self._update_U(d2)
            J = self._objective(U_new, d2)
            history.append(J)
            du = np.abs(U_new - U).max()
            dJ = abs(history[-2] - J) / max(abs(history[-2]), 1e-300) if len(history) > 1 else np.inf
            U = U_new
            if du < self.tol or dJ < self.tol_obj:
                converged = True
                break
        V = self._update_centers(X, U)
        d2 = squared_distance(X, V, A)
        return _RunResult(
            U=U, V=V, n_iter=k, converged=converged,
            history=np.asarray(history), objective=self._objective(U, d2),
        )

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples < n_clusters")
        A = self._resolve_A(X)
        rng = np.random.default_rng(self.random_state)

        best = self._single_run(X, A, rng)
        for _ in range(max(1, self.n_init) - 1):
            run = self._single_run(X, A, rng)
            if run.objective < best.objective:
                best = run

        self.U_ = best.U
        self.memberships_ = best.U.T
        self.centers_ = best.V
        self.n_iter_ = best.n_iter
        self.converged_ = best.converged
        self.objective_ = best.objective
        self.objective_history_ = best.history
        self.A_ = A
        return self

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def transform(self, X):
        """Fuzzy membership of new points to the fitted prototypes, (n, c)."""
        X = np.asarray(X, dtype=float)
        d2 = squared_distance(X, self.centers_, self.A_)
        return self._update_U(d2).T

    def predict(self, X):
        """Hard labels via maximum membership (defuzzification by argmax)."""
        return self.transform(X).argmax(axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.U_.argmax(axis=0)

    def boundary_distortion_rate(self, U=None):
        """Fraction of samples whose top membership is < 0.5 (ambiguous assignments).

        This is the quantity contrasted against crisp Voronoi partitions in Module 2.3.
        """
        U = self.U_ if U is None else np.asarray(U, dtype=float)
        return float(np.mean(U.max(axis=0) < 0.5))


if __name__ == "__main__":  # smoke test
    from .data import factor_scores, make_synthetic_ipip
    from .validity import summary

    X_items, y_true, _ = make_synthetic_ipip(n=1500, c=5, seed=0)
    X = factor_scores(X_items)          # (n, 5) standardized OCEAN scores — Module 3
    eng = FuzzyCMeansEngine(n_clusters=5, m=2.0, tol=1e-5, random_state=0).fit(X)

    hist = eng.objective_history_
    monotone = bool(np.all(np.diff(hist) <= 1e-6))
    col_err = float(np.abs(eng.U_.sum(axis=0) - 1.0).max())

    print(f"converged      : {eng.converged_}  (n_iter={eng.n_iter_} / {eng.max_iter})")
    print(f"J_m            : {hist[0]:.3f} -> {hist[-1]:.3f}   monotone non-increasing: {monotone}")
    print(f"max|sum_i u-1| : {col_err:.2e}")
    print(f"boundary rate  : {eng.boundary_distortion_rate():.3f}")
    print("validity       :", {k: round(v, 4) for k, v in summary(X, eng.U_, eng.centers_, eng.m).items()})

    # m -> 1+ must approach a crisp partition (Module 1.4)
    crisp = FuzzyCMeansEngine(n_clusters=5, m=1.05, random_state=0).fit(X)
    print(f"m=1.05 mean max-membership: {crisp.U_.max(axis=0).mean():.3f} (expect ~1.0)")
