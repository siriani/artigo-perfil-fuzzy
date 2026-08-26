"""Shared setup for the experiment scripts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

# One fixed configuration for every experiment, so tables/figures are consistent.
N_SUBSAMPLE = 25_000
SEED = 42


def load_dataset(space="factor", n=N_SUBSAMPLE, seed=SEED, verbose=True):
    """Return (X, meta). space in {"factor", "pca", "items"}.

    Samples ``n`` respondents at random from the *entire* cleaned IPIP-50 population
    (complete responses, IPC == 1). Falls back to synthetic data (with a warning) if
    the real dump is not present.
    """
    from fuzzyprofile.data import load_ipip50, make_synthetic_ipip
    from fuzzyprofile.preprocess import covariance_diagnostics, factor_scores, spherical_pca

    n_available = None
    try:
        items, n_available = load_ipip50(ipc_unique=True, sample=n, random_state=seed,
                                         return_n_available=True)
        source = "IPIP-50 (openpsychometrics, IPC==1)"
    except FileNotFoundError:
        items, _, _ = make_synthetic_ipip(n=n, c=5, seed=seed)
        source = "SYNTHETIC (real dump not downloaded — run scripts/get_data.sh)"

    pca_k = None
    if space == "items":
        X = items - items.mean(axis=0)
    elif space == "pca":
        X, info = spherical_pca(items, var_target=0.95)
        pca_k = int(info["k"])
    else:
        X = factor_scores(items)

    meta = {"source": source, "n": len(X), "n_available": n_available,
            "d": X.shape[1], "space": space, "pca_k": pca_k,
            "cov": covariance_diagnostics(X)}
    if verbose:
        na = f"{n_available:,}" if n_available else "n/a"
        print(f"[data] {source} | X={X.shape} | space={space} | "
              f"n_available={na} | cov.cond={meta['cov']['cond']:.1f} "
              f"mahalanobis_safe={meta['cov']['mahalanobis_safe']}")
    return np.asarray(X, dtype=float), meta
