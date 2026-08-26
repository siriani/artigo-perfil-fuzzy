"""Module 2.2 — grid optimization of (c, m) on the IPIP-50 OCEAN-5 space.

Sweeps c in [2, 10], m in [1.1, 3.0] step 0.1 for the Euclidean metric (and Mahalanobis
if the covariance is well-conditioned). Writes:
  results/gridsearch_<metric>.csv
  paper/figures/fig_cost_surface.png
and prints the (c*, m*) picks.
"""
from __future__ import annotations

import json

from _common import RESULTS, SEED, load_dataset

from fuzzyprofile.gridsearch import grid_search, optimal_pair


def main():
    X, meta = load_dataset(space="factor")
    print(f"[grid] n={meta['n']} d={meta['d']}  ({meta['source']})\n")

    picks = {}
    for metric in ("euclidean", "mahalanobis"):
        if metric == "mahalanobis" and not meta["cov"]["mahalanobis_safe"]:
            print("[grid] skipping Mahalanobis (ill-conditioned covariance)")
            continue
        print(f"--- metric = {metric} ---")
        df = grid_search(X, metric=metric, n_init=3, random_state=SEED, verbose=True)
        csv = RESULTS / f"gridsearch_{metric}.csv"
        df.to_csv(csv, index=False)
        opt = optimal_pair(df, primary="XB")
        picks[metric] = {
            "c_star": opt["c_star"], "m_star": opt["m_star"],
            "votes": opt["votes_at_primary_optimum"],
            "index_optima": {k: list(v) for k, v in opt["index_optima"].items()},
            "marginal_best_c": opt["marginal_best_c"],
        }
        print(f"\n[grid:{metric}] (c*, m*) = ({opt['c_star']}, {opt['m_star']:.1f}) "
              f"by XB | consensus votes = {opt['votes_at_primary_optimum']}/6")
        print(f"[grid:{metric}] per-index optima: {opt['index_optima']}")
        print(f"[grid:{metric}] marginal best c: {opt['marginal_best_c']}")
        print(f"[grid:{metric}] wrote {csv}\n")

    (RESULTS / "gridsearch_picks.json").write_text(json.dumps(picks, indent=2))

    try:
        import pandas as pd

        import figures
        df_e = pd.read_csv(RESULTS / "gridsearch_euclidean.csv")
        fig = figures.cost_surfaces(df_e, indices=("XB", "FPC", "PE"))
        print(f"[grid] wrote {fig}")
    except Exception as e:  # pragma: no cover
        print(f"[grid] figure skipped: {e}")


if __name__ == "__main__":
    main()
