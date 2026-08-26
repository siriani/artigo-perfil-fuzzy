"""Hyperparameter grid optimization for FCM (Module 2.2).

Systematic sweep of c in [2, 10] and m in [1.1, 3.0] (step 0.1). For every (c, m):
fit FuzzyCMeansEngine, record FPC, MPC, PE, Xie-Beni, Fukuyama-Sugeno, Kwon, the
boundary distortion rate, iterations, convergence flag and wall time.

``optimal_pair`` reads (c*, m*) off the validity surfaces; ``cost_surface`` returns the
meshgrids for contour/heatmap plots used in the paper.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .engine import FuzzyCMeansEngine
from .validity import summary

DEFAULT_C = tuple(range(2, 11))
DEFAULT_M = tuple(round(x, 2) for x in np.arange(1.1, 3.0 + 1e-9, 0.1))

# Direction of improvement for each recorded index.
MAXIMIZE = {"FPC", "MPC"}
MINIMIZE = {"PE", "XB", "FS", "Kwon"}


def grid_search(X, c_values=DEFAULT_C, m_values=DEFAULT_M, metric="euclidean",
                A=None, n_init=2, max_iter=300, tol=1e-4, random_state=0, verbose=True):
    """Run FCM over the (c, m) grid. Returns a tidy DataFrame, one row per pair.

    ``tol`` is on ``||Delta U||_inf``; the engine also stops on a relative objective
    change below ``tol_obj`` (1e-7), which is what converges the weakly-clustered
    real-data cells.
    """
    X = np.asarray(X, dtype=float)
    rows = []
    total = len(c_values) * len(m_values)
    k = 0
    c_list, m_list = list(c_values), list(m_values)
    for c in c_list:
        for m in m_list:
            k += 1
            t0 = time.perf_counter()
            eng = FuzzyCMeansEngine(
                n_clusters=int(c), m=float(m), metric=metric, A=A,
                n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state,
            ).fit(X)
            dt = time.perf_counter() - t0
            s = summary(X, eng.U_, eng.centers_, eng.m)
            rows.append({
                "c": int(c), "m": float(m), "metric": metric,
                **{key: float(val) for key, val in s.items()},
                "boundary_rate": eng.boundary_distortion_rate(),
                "n_iter": int(eng.n_iter_), "converged": bool(eng.converged_),
                "objective": float(eng.objective_), "seconds": dt,
            })
            if verbose and (k % 20 == 0 or k == total):
                print(f"  [{k:3d}/{total}] c={c} m={m:.1f}  "
                      f"FPC={s['FPC']:.3f} XB={s['XB']:.4f}  ({dt*1e3:.0f} ms)")
    return pd.DataFrame(rows)


def optimal_pair(df, primary="XB"):
    """Best (c, m) by ``primary`` index, with a consensus vote across all indices.

    Returns a dict: {primary, c_star, m_star, row, votes} where ``votes`` counts how
    many indices place their optimum at (c_star, m_star).
    """
    df = df.reset_index(drop=True)
    best_idx = df[primary].idxmin() if primary in MINIMIZE else df[primary].idxmax()
    row = df.loc[best_idx]
    c_star, m_star = int(row["c"]), float(row["m"])

    votes = 0
    picks = {}
    for idx in MAXIMIZE | MINIMIZE:
        if idx not in df.columns:
            continue
        j = df[idx].idxmin() if idx in MINIMIZE else df[idx].idxmax()
        picks[idx] = (int(df.loc[j, "c"]), round(float(df.loc[j, "m"]), 2))
        if picks[idx] == (c_star, round(m_star, 2)):
            votes += 1

    # marginal best c: index optimum after averaging over m
    by_c = df.groupby("c")[list(MAXIMIZE | MINIMIZE)].mean()
    c_consensus = {
        idx: int(by_c[idx].idxmax() if idx in MAXIMIZE else by_c[idx].idxmin())
        for idx in by_c.columns
    }
    return {
        "primary": primary,
        "c_star": c_star, "m_star": m_star,
        "row": row.to_dict(),
        "index_optima": picks,
        "votes_at_primary_optimum": votes,
        "marginal_best_c": c_consensus,
    }


def cost_surface(df, index):
    """Meshgrids (C, M, Z) of ``index`` over the grid, for heatmaps/contours."""
    piv = df.pivot(index="c", columns="m", values=index).sort_index().sort_index(axis=1)
    C, M = np.meshgrid(piv.columns.to_numpy(), piv.index.to_numpy())
    return M, C, piv.to_numpy()  # (c-mesh, m-mesh, Z)


if __name__ == "__main__":
    from .data import make_synthetic_ipip
    from .preprocess import factor_scores

    Xi, _, _ = make_synthetic_ipip(n=1500, c=5, seed=0)
    Z = factor_scores(Xi)
    df = grid_search(Z, c_values=range(2, 8), m_values=(1.2, 1.5, 2.0, 2.5),
                     n_init=2, random_state=0)
    print(df.round(3).to_string(index=False))
    print()
    opt = optimal_pair(df, primary="XB")
    print(f"(c*, m*) = ({opt['c_star']}, {opt['m_star']})  "
          f"votes={opt['votes_at_primary_optimum']}  marginal_best_c={opt['marginal_best_c']}")
