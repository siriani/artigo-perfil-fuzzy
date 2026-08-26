"""End-to-end pipeline (Modules 2.2 + 2.3 + 3), single data load per feature space.

    python experiments/run_all.py

Writes results/{gridsearch_*.csv, cvi_by_c_*.csv, rho_profiles.json, benchmark.csv,
gmm_by_c.csv, summary.json} and paper/figures/*.png.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from _common import RESULTS, SEED, load_dataset

from fuzzyprofile import FuzzyCMeansEngine
from fuzzyprofile.benchmarks import (
    boundary_rate, compare, gaussian_mixture, gustafson_kessel, hard_c_means,
    possibilistic_c_means,
)
from fuzzyprofile.data import make_synthetic_ipip
from fuzzyprofile.gridsearch import grid_search, optimal_pair
from fuzzyprofile.preprocess import covariance_diagnostics, factor_scores, spherical_pca
from fuzzyprofile.validity import partition_entropy
from fuzzyprofile.validity import summary as cvi_summary

C_GRID = range(2, 11)
M_OP = 2.0            # ponto de operação (fuzzifier padrão)
C_FIXED = 6          # c do ponto de operação e das curvas rho(m)
M_PROFILE = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.75, 2.0, 2.25, 2.5]


def _r(o, k=4):
    if isinstance(o, dict):
        return {kk: _r(v, k) for kk, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_r(v, k) for v in o]
    if isinstance(o, (float, np.floating)):
        return round(float(o), k)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def cvi_by_c(X, m, metric="euclidean", A=None):
    rows = []
    for c in C_GRID:
        e = FuzzyCMeansEngine(n_clusters=c, m=m, metric=metric, A=A,
                              n_init=3, random_state=SEED).fit(X)
        s = cvi_summary(X, e.U_, e.centers_, e.m)
        rows.append({"c": c, **{kk: round(v, 4) for kk, v in s.items()},
                     "boundary_rate": round(e.boundary_distortion_rate(), 4),
                     "n_iter": int(e.n_iter_), "converged": bool(e.converged_)})
    return pd.DataFrame(rows)


def rho_profile(X, c=C_FIXED, metric="euclidean", A=None):
    out = {"m": M_PROFILE, "rho": [], "pe_norm": [], "mean_max": []}
    for m in M_PROFILE:
        e = FuzzyCMeansEngine(n_clusters=c, m=m, metric=metric, A=A,
                              n_init=3, random_state=SEED).fit(X)
        out["rho"].append(round(e.boundary_distortion_rate(), 4))
        out["pe_norm"].append(round(float(partition_entropy(e.U_) / np.log(c)), 4))
        out["mean_max"].append(round(float(e.U_.max(axis=0).mean()), 4))
    return out


def gmm_by_c(X):
    """GMM route (Gerlach): responsibilities, posterior uncertainty, BIC vs c."""
    rows = []
    for c in C_GRID:
        gm = gaussian_mixture(X, c, random_state=SEED)
        U = gm["U"]
        rows.append({
            "c": c,
            "rho_gmm": round(boundary_rate(U), 4),         # frac. max posterior < 0.5
            "mean_max_post": round(float(np.mean(U.max(axis=0))), 4),
            "bic": round(gm["bic"], 1), "aic": round(gm["aic"], 1),
            "converged": gm["converged"],
        })
    return pd.DataFrame(rows)


def main():
    t0 = time.perf_counter()
    summary = {}

    # ---- feature spaces --------------------------------------------------
    Xf, meta = load_dataset(space="factor")
    items, _ = load_dataset(space="items", verbose=False)
    Xp, meta_p = load_dataset(space="pca", verbose=False)
    _, pca_info = spherical_pca(items + items.mean(0), var_target=0.95)

    summary["data"] = _r({
        "source": meta["source"], "n": meta["n"],
        "n_available_ipc1": meta["n_available"],
        "cov_factor": covariance_diagnostics(Xf),
        "cov_items_d50": covariance_diagnostics(items),
        "pca_k_for_95pct": int(pca_info["k"]),
    }, 3)
    print("[diag]", json.dumps(summary["data"], indent=2), "\n")

    # ---- Module 2.2: full grid on OCEAN-5 Euclidean -----------------
    print("[grid] OCEAN-5 Euclidean, c=2..10, m=1.1..3.0 ...")
    dfg = grid_search(Xf, metric="euclidean", n_init=2, random_state=SEED, verbose=True)
    dfg.to_csv(RESULTS / "gridsearch_euclidean.csv", index=False)
    opt = optimal_pair(dfg, primary="XB")
    c_star, m_star = opt["c_star"], opt["m_star"]
    print(f"[grid] (c*, m*)_XB = ({c_star}, {m_star:.1f}); consenso "
          f"{opt['votes_at_primary_optimum']}/6; per-index {opt['index_optima']}\n")

    # ---- CVI vs c and rho(m) for the three spaces ------------------
    from fuzzyprofile.distances import covariance_metric
    A_mahal = covariance_metric(Xf)
    spaces = {
        "ocean5_euclid": dict(X=Xf, metric="euclidean", A=None, d=Xf.shape[1]),
        "ocean5_mahal": dict(X=Xf, metric="mahalanobis", A=A_mahal, d=Xf.shape[1]),
        "pca_spherical": dict(X=Xp, metric="euclidean", A=None, d=Xp.shape[1]),
    }
    summary["spaces"] = {}
    for name, sp in spaces.items():
        dcvi = cvi_by_c(sp["X"], M_OP, metric=sp["metric"], A=sp["A"])
        dcvi.to_csv(RESULTS / f"cvi_by_c_{name}.csv", index=False)
        rp = rho_profile(sp["X"], metric=sp["metric"], A=sp["A"])
        summary["spaces"][name] = _r({
            "d": sp["d"],
            "cvi_by_c_at_mop": dcvi.set_index("c").to_dict(orient="index"),
            "rho_profile_c6": rp,
        }, 4)
        print(f"[{name}] rho(m) c={C_FIXED}: "
              + ", ".join(f"{m}:{r}" for m, r in zip(rp["m"], rp["rho"])))

    # raw-50 control
    d50 = cvi_by_c(items - items.mean(0), M_OP)
    summary["raw50_control"] = _r(d50.set_index("c").to_dict(orient="index"), 4)

    # ---- synthetic calibration (M4): data WITH real 5-cluster structure ----
    Xs = factor_scores(make_synthetic_ipip(n=meta["n"], c=5, seed=SEED)[0])
    rp_syn = rho_profile(Xs, c=5)
    summary["synthetic_calibration"] = _r({
        "note": "5 perfis OCEAN separáveis; mesma n",
        "rho_profile_c5": rp_syn,
        "cvi_by_c_at_mop": cvi_by_c(Xs, M_OP).set_index("c").to_dict(orient="index"),
    }, 4)
    print(f"[synthetic] rho(m) c=5: "
          + ", ".join(f"{m}:{r}" for m, r in zip(rp_syn["m"], rp_syn["rho"])))

    # ---- Module 2.3: benchmark (+GMM) at (c*, m*) on OCEAN-5 --------
    print(f"\n[bench] FCM vs Hard/GK/PCM/GMM at (c, m) = ({C_FIXED}, {M_OP}) ...")
    dfb = compare(Xf, c=C_FIXED, m=M_OP, random_state=SEED)
    dfb.to_csv(RESULTS / "benchmark.csv", index=False)
    print(dfb.round(4).to_string(index=False), "\n")

    dgmm = gmm_by_c(Xf)
    dgmm.to_csv(RESULTS / "gmm_by_c.csv", index=False)
    print("[gmm] posterior uncertainty vs c:\n", dgmm.to_string(index=False), "\n")

    # ---- figures --------------------------------------------------
    U_by = {
        "FCM": FuzzyCMeansEngine(C_FIXED, m=M_OP, n_init=5, random_state=SEED).fit(Xf).U_,
        "Hard c-means": hard_c_means(Xf, C_FIXED, random_state=SEED)["U"],
        "Gustafson-Kessel": gustafson_kessel(Xf, C_FIXED, m=M_OP, n_init=2, random_state=SEED)["U"],
        "PCM": possibilistic_c_means(Xf, C_FIXED, m=M_OP, random_state=SEED)["U"],
        "GMM": gaussian_mixture(Xf, C_FIXED, random_state=SEED)["U"],
    }
    hist, labs = [], []
    for c, m in [(C_FIXED, 1.2), (C_FIXED, 1.5), (C_FIXED, 2.0), (9, 1.5), (2, 1.5)]:
        e = FuzzyCMeansEngine(c, m=m, n_init=1, random_state=SEED).fit(Xf)
        hist.append(e.objective_history_); labs.append(f"c={c}, m={m}")
    try:
        import figures
        rp0 = summary["spaces"]["ocean5_euclid"]["rho_profile_c6"]
        for f in (
            figures.pipeline(),
            figures.cost_surfaces(dfg, indices=("XB", "FPC", "PE")),
            figures.convergence(hist, labs),
            figures.benchmark_bars(dfb),
            figures.membership_hist(U_by),
            figures.rho_vs_m(rp0["m"], rp0["rho"], rp0["pe_norm"], rp0["mean_max"], C_FIXED),
            figures.rho_calibration(rp0, rp_syn, dgmm),
        ):
            print("[fig]", f)
    except Exception as e:  # pragma: no cover
        import traceback; traceback.print_exc()
        print(f"[fig] skipped: {e}")

    # ---- summary.json ------------------------------------------
    star_row = dfg[(dfg.c == c_star) & (np.isclose(dfg.m, m_star))].iloc[0]
    e_op = FuzzyCMeansEngine(C_FIXED, m=M_OP, n_init=5, random_state=SEED).fit(Xf)
    summary["grid_euclid"] = _r({
        "c_star_xb": c_star, "m_star_xb": m_star,
        "consensus_votes_out_of_6": opt["votes_at_primary_optimum"],
        "index_optima": {k: list(v) for k, v in opt["index_optima"].items()},
        "marginal_best_c": opt["marginal_best_c"],
        "at_xb_optimum": star_row.to_dict(),
        "n_cells": len(dfg), "n_converged": int(dfg.converged.sum()),
        "median_iter": float(dfg.n_iter.median()),
    }, 5)
    summary["operating_point"] = _r({
        "c": C_FIXED, "m": M_OP,
        "fcm": {"n_iter": int(e_op.n_iter_), "converged": bool(e_op.converged_),
                "monotone_Jm": bool(np.all(np.diff(e_op.objective_history_) <= 1e-6)),
                "boundary_rate": e_op.boundary_distortion_rate(),
                "mean_max_membership": float(e_op.U_.max(axis=0).mean())},
        "benchmark": dfb.set_index("method").to_dict(orient="index"),
        "gmm_by_c": dgmm.set_index("c").to_dict(orient="index"),
    }, 4)
    summary["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("=" * 62)
    print(f"[all] done in {summary['runtime_seconds']:.0f}s -> results/summary.json")


if __name__ == "__main__":
    main()
