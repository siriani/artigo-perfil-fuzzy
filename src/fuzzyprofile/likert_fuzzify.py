"""Uncertainty modeling of discrete Likert responses (Module 1.4).

A Likert answer r in {1,...,5} is not a point on a ratio scale: it carries epistemic
uncertainty (the respondent's hesitation) and coarse quantization. Two encodings:

Triangular Fuzzy Number (TFN)
    r  ->  (a, b, c) = (r - delta, r, r + delta), clipped to [x_min, x_max].
    Defuzzification (centroid):  x_hat = (a + b + c) / 3.
    Spread s = (c - a) / 2 quantifies the retained ambiguity.

Intuitionistic Fuzzy Set (IFS)
    r  ->  (mu, nu, pi) with mu + nu <= 1, pi = 1 - mu - nu the hesitation margin.
    Here mu is the normalized agreement (r-1)/(x_max-1), nu a reluctance term, and
    pi grows toward the centre of the scale where responses are least committal.

Only the TFN encoder + centroid defuzzification are implemented now; the IFS encoder
returns the (mu, nu, pi) triples for downstream score-function experiments.
"""
from __future__ import annotations

import numpy as np

LIKERT_MIN, LIKERT_MAX = 1, 5


def triangular_fuzzify(R, delta=0.5, x_min=LIKERT_MIN, x_max=LIKERT_MAX):
    """Map an array of Likert responses to TFN triples.

    Parameters
    ----------
    R : array_like, integer responses in [x_min, x_max]
    delta : float or array_like
        Half-width of the triangular support. Scalar, or per-item / per-response.

    Returns
    -------
    tfn : ndarray (..., 3)
        Stacked (a, b, c) with a <= b <= c, clipped to [x_min, x_max].
    """
    R = np.asarray(R, dtype=float)
    delta = np.asarray(delta, dtype=float)
    a = np.clip(R - delta, x_min, x_max)
    b = np.clip(R, x_min, x_max)
    c = np.clip(R + delta, x_min, x_max)
    return np.stack([a, b, c], axis=-1)


def defuzzify_centroid(tfn):
    """Centroid defuzzification of TFN triples: x_hat = (a + b + c) / 3."""
    tfn = np.asarray(tfn, dtype=float)
    return tfn.mean(axis=-1)


def tfn_spread(tfn):
    """Ambiguity retained per response: s = (c - a) / 2."""
    tfn = np.asarray(tfn, dtype=float)
    return (tfn[..., 2] - tfn[..., 0]) / 2.0


def tfn_distance(p, q):
    """Vertex (L2 over (a,b,c)) distance between two TFN arrays — a valid metric on TFNs."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return np.sqrt(np.sum((p - q) ** 2, axis=-1))


def intuitionistic_fuzzify(R, x_min=LIKERT_MIN, x_max=LIKERT_MAX, reluctance=0.15):
    """Map Likert responses to IFS triples (mu, nu, pi), mu + nu <= 1.

    mu  = (r - x_min) / (x_max - x_min)                 agreement
    nu  = reluctance * (1 - |r - mid| / (mid - x_min))  reluctance, peaks at scale centre
    pi  = 1 - mu - nu                                   hesitation margin
    """
    R = np.asarray(R, dtype=float)
    span = x_max - x_min
    mid = 0.5 * (x_max + x_min)
    mu = (R - x_min) / span
    nu = reluctance * (1.0 - np.abs(R - mid) / (mid - x_min))
    nu = np.clip(nu, 0.0, 1.0 - mu)
    pi = 1.0 - mu - nu
    return np.stack([mu, nu, pi], axis=-1)


def defuzzify_ifs(ifs, x_min=LIKERT_MIN, x_max=LIKERT_MAX):
    """Point estimate from an IFS triple (mu, nu, pi).

    The mu-implied value  x_mu = x_min + mu * span  is shrunk toward the scale centre
    in proportion to the reluctance nu:  x_hat = (1 - nu) * x_mu + nu * mid.
    With nu = 0 this recovers the raw response; nu > 0 pulls uncertain (mid-scale)
    responses toward neutral.
    """
    ifs = np.asarray(ifs, dtype=float)
    mu, nu = ifs[..., 0], ifs[..., 1]
    span = x_max - x_min
    mid = 0.5 * (x_max + x_min)
    x_mu = x_min + mu * span
    return (1.0 - nu) * x_mu + nu * mid


def fuzzify_matrix(X, method="tfn", **kw):
    """Encode an (n, d) Likert matrix, then defuzzify back to (n, d) for projection.

    method="none" -> passthrough (raw responses)
    method="tfn"  -> centroid of (r-delta, r, r+delta)  [kw: delta]
    method="ifs"  -> reluctance-shrunk point estimate    [kw: reluctance]
    """
    X = np.asarray(X, dtype=float)
    if method == "none":
        return X
    if method == "tfn":
        return defuzzify_centroid(triangular_fuzzify(X, **kw))
    if method == "ifs":
        return defuzzify_ifs(intuitionistic_fuzzify(X, **kw))
    raise ValueError(f"unknown method {method!r}")


if __name__ == "__main__":
    r = np.array([1, 2, 3, 4, 5])
    tfn = triangular_fuzzify(r, delta=0.5)
    print("TFN triples:\n", tfn)
    print("centroid   :", defuzzify_centroid(tfn))
    print("spread     :", tfn_spread(tfn))
    print("IFS triples:\n", np.round(intuitionistic_fuzzify(r), 3))
