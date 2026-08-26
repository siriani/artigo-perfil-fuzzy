"""IPIP-50 (Big Five / OCEAN) data access: real loader + synthetic generator.

Real data
---------
Open-Source Psychometrics Project dump = Kaggle ``tunguz/big-five-personality-test``.
File ``IPIP-FFM-data-8Nov2018/data-final.csv`` (TAB-separated, ~1.0e6 rows).
50 Likert items 1..5 in blocks EXT1..10, EST1..10, AGR1..10, CSN1..10, OPN1..10
(EST = Neuroticism). Fetch with ``scripts/get_data.sh``.

Synthetic fallback
------------------
``make_synthetic_ipip`` draws ``c`` latent OCEAN profiles and emits ordinal 1..5
responses with per-item polarity + Gaussian noise, so the pipeline runs with no
download and with a known ground truth.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

FACTORS = ("EXT", "EST", "AGR", "CSN", "OPN")          # O C E A N (EST = Neuroticism)
ITEMS_PER_FACTOR = 10
LIKERT_MIN, LIKERT_MAX = 1, 5

# Reverse-keyed items (1-indexed within each factor block) — IPIP-50 standard key.
REVERSE_KEYED = {
    "EXT": (2, 4, 6, 8, 10),
    "EST": (2, 4),
    "AGR": (1, 3, 5, 7),
    "CSN": (2, 4, 6, 8),
    "OPN": (2, 4, 6),
}

DEFAULT_PATHS = (
    "data/raw/IPIP-FFM-data-8Nov2018/data-final.csv",
    "data/raw/IPIP-FFM-data-8Nov2018/data.csv",
    "data/raw/data-final.csv",
    "data/raw/data.csv",
)


def item_columns():
    """The 50 item names in canonical order."""
    return [f"{f}{i}" for f in FACTORS for i in range(1, ITEMS_PER_FACTOR + 1)]


def reverse_keyed_columns():
    return [f"{f}{i}" for f, idxs in REVERSE_KEYED.items() for i in idxs]


def reverse_score(df, cols=None):
    """Polarity inversion of reverse-keyed items:  x' = (x_max + x_min) - x = 6 - x."""
    df = df.copy()
    cols = [c for c in (cols or reverse_keyed_columns()) if c in df.columns]
    df[cols] = (LIKERT_MAX + LIKERT_MIN) - df[cols]
    return df


def factor_scores(X, standardize=True):
    """Reduce an (n, 50) trait-aligned item matrix to (n, 5) OCEAN factor scores.

    Each score is the mean of its 10 items. Direct projection to the 5-D personality
    space (Module 3): FCM on the 50 raw items loses all distance contrast in high
    dimension, so profiling is done on the factor scores (or PCA components).
    """
    X = np.asarray(X, dtype=float)
    if X.shape[1] != len(FACTORS) * ITEMS_PER_FACTOR:
        raise ValueError(f"expected {len(FACTORS)*ITEMS_PER_FACTOR} item columns, got {X.shape[1]}")
    blocks = X.reshape(X.shape[0], len(FACTORS), ITEMS_PER_FACTOR)
    S = blocks.mean(axis=2)                                   # (n, 5)
    if standardize:
        S = (S - S.mean(axis=0)) / np.maximum(S.std(axis=0), np.finfo(float).tiny)
    return S


def load_ipip50(path=None, apply_reverse=True, dropna=True, max_rows=None,
                ipc_unique=True, sample=None, random_state=0, as_frame=False,
                return_n_available=False):
    """Load the real IPIP-50 responses as an (n, 50) array (or DataFrame).

    Invalid / unanswered items (value 0 or outside 1..5) become NaN. With
    ``dropna=True`` rows containing any NaN are removed.

    Parameters
    ----------
    max_rows : int or None
        If given, read only the first ``max_rows`` lines (fast path). If None, the
        whole file is read (only the 50 item columns + IPC are loaded, so ~1e6 rows
        cost <1 GB and a few seconds).
    ipc_unique : bool
        Keep only records with ``IPC == 1`` (one submission per IP) if the column is
        present -- the "max cleanliness" filter recommended in the dataset codebook.
    sample : int or None
        After cleaning, draw a random subsample of this size (without replacement),
        from the *whole* cleaned population.
    random_state : int
        Seed for ``sample``.
    return_n_available : bool
        If True, also return the number of clean records before subsampling.

    Raises
    ------
    FileNotFoundError
        If no data file is found. Run ``scripts/get_data.sh`` or use
        :func:`make_synthetic_ipip`.
    """
    candidates = [path] if path else list(DEFAULT_PATHS)
    src = next((p for p in candidates if p and os.path.exists(p)), None)
    if src is None:
        raise FileNotFoundError(
            "IPIP-50 data file not found. Run `bash scripts/get_data.sh` "
            "or use make_synthetic_ipip()."
        )

    wanted = item_columns() + ["IPC"]
    header = pd.read_csv(src, sep="\t", nrows=0).columns
    usecols = [c for c in wanted if c in header]
    df = pd.read_csv(src, sep="\t", nrows=max_rows, usecols=usecols, low_memory=False)
    cols = [c for c in item_columns() if c in df.columns]
    if len(cols) != 50:
        raise ValueError(
            f"expected 50 IPIP item columns in {src}, found {len(cols)}. "
            f"First few present: {cols[:6]}"
        )

    if ipc_unique and "IPC" in df.columns:
        df = df[pd.to_numeric(df["IPC"], errors="coerce") == 1]

    data = df[cols].apply(pd.to_numeric, errors="coerce")
    data = data.where((data >= LIKERT_MIN) & (data <= LIKERT_MAX))
    if dropna:
        data = data.dropna(axis=0)
    if apply_reverse:
        data = reverse_score(data)
    data = data.reset_index(drop=True)
    n_available = len(data)

    if sample is not None and sample < len(data):
        data = data.sample(n=sample, random_state=random_state).reset_index(drop=True)

    out = data if as_frame else data.to_numpy(dtype=float)
    return (out, n_available) if return_n_available else out


def _spread_profiles(rng, c, d, lo=1.8, hi=4.2, min_sep=2.0, tries=4000):
    """Draw ``c`` latent profiles in [lo, hi]^d with pairwise L2 separation >= min_sep.

    Rejection sampling; the separation floor is relaxed if ``c`` points cannot be placed.
    """
    P = rng.uniform(lo, hi, size=(1, d))
    for _ in range(tries):
        if P.shape[0] >= c:
            break
        cand = rng.uniform(lo, hi, size=(1, d))
        if float(np.min(np.linalg.norm(P - cand, axis=1))) >= min_sep:
            P = np.vstack([P, cand])
    while P.shape[0] < c:                        # relax: accept whatever is left
        P = np.vstack([P, rng.uniform(lo, hi, size=(1, d))])
    return P[:c]


def make_synthetic_ipip(n=2000, c=5, seed=0, noise=0.4, item_bias=0.15,
                        min_sep=2.0, as_frame=False):
    """Synthetic IPIP-like responses from ``c`` separable latent OCEAN profiles.

    ``noise`` is the per-item response SD on the 1..5 scale; ``min_sep`` the minimum
    L2 distance between latent profiles (5-D OCEAN space). Defaults give clusters that
    are recoverable but overlapping at the boundaries.

    Returns
    -------
    X : ndarray (n, 50)  (or DataFrame if ``as_frame``)
        Reverse-scored, trait-aligned responses on the 1..5 scale (float).
    y : ndarray (n,)
        Ground-truth latent profile index in [0, c).
    meta : dict
        {"profiles": (c, 5) latent factor means, "factors": FACTORS,
         "reverse_keyed": REVERSE_KEYED}
    """
    rng = np.random.default_rng(seed)
    profiles = _spread_profiles(rng, c, len(FACTORS), min_sep=min_sep)   # latent OCEAN means
    y = rng.integers(0, c, size=n)
    cols = item_columns()
    X = np.empty((n, len(cols)), dtype=float)

    for f_i, f in enumerate(FACTORS):
        base = profiles[y, f_i]                                # (n,)
        rev = set(REVERSE_KEYED.get(f, ()))
        for it in range(1, ITEMS_PER_FACTOR + 1):
            col = f_i * ITEMS_PER_FACTOR + (it - 1)
            latent = base + rng.normal(0, item_bias) + rng.normal(0, noise, size=n)
            resp = np.clip(np.rint(latent), LIKERT_MIN, LIKERT_MAX)
            if it in rev:                                      # store as raw questionnaire
                resp = (LIKERT_MAX + LIKERT_MIN) - resp
            X[:, col] = resp

    df = reverse_score(pd.DataFrame(X, columns=cols))          # recover trait alignment
    meta = {"profiles": profiles, "factors": FACTORS, "reverse_keyed": REVERSE_KEYED}
    if as_frame:
        return df, y, meta
    return df.to_numpy(dtype=float), y, meta


if __name__ == "__main__":
    X, y, meta = make_synthetic_ipip(n=8, c=3, seed=1)
    print("synthetic X shape:", X.shape, "| y:", y)
    print("latent profiles (c x OCEAN):\n", np.round(meta["profiles"], 2))
    try:
        real = load_ipip50(max_rows=1000)
        print("real IPIP-50 sample:", real.shape)
    except FileNotFoundError as e:
        print("real data not downloaded:", e)
