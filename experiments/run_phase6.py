"""Fase 6: (A) ablação da codificação Likert TFN/IFS, (B) estabilidade sob
reamostragem, (C) interpretação dos protótipos c=6.

    python experiments/run_phase6.py

Escreve results/phase6.json e paper/figures/{fig_ablation.png, fig_prototypes.png}.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from _common import FIGDIR, RESULTS, SEED, load_dataset

from fuzzyprofile import FuzzyCMeansEngine
from fuzzyprofile.data import FACTORS, factor_scores, load_ipip50, make_synthetic_ipip
from fuzzyprofile.gridsearch import grid_search, optimal_pair
from fuzzyprofile.likert_fuzzify import fuzzify_matrix
from fuzzyprofile.validity import partition_entropy
from fuzzyprofile.validity import summary as cvi_summary

C_FIXED, M_OP = 6, 2.0
M_RED = (1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5)          # grade reduzida
M_PROFILE = [1.1, 1.3, 1.5, 1.75, 2.0, 2.5]


def _round(o, k=4):
    if isinstance(o, dict):
        return {kk: _round(v, k) for kk, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, k) for v in o]
    if isinstance(o, (float, np.floating)):
        return round(float(o), k)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def rho_at(X, c, m):
    e = FuzzyCMeansEngine(c, m=m, n_init=2, random_state=SEED).fit(X)
    return e.boundary_distortion_rate()


def fpc_at(X, c, m):
    e = FuzzyCMeansEngine(c, m=m, n_init=2, random_state=SEED).fit(X)
    return cvi_summary(X, e.U_, e.centers_, m)["FPC"]


# ---------------------------------------------------------------- (A) ablação
def ablation(items):
    encs = {
        "raw":       dict(method="none"),
        "tfn_d0.5":  dict(method="tfn", delta=0.5),
        "tfn_d1.0":  dict(method="tfn", delta=1.0),
        "ifs_r0.15": dict(method="ifs", reluctance=0.15),
        "ifs_r0.30": dict(method="ifs", reluctance=0.30),
    }
    rows, rho_curves = [], {}
    ref = factor_scores(items)                       # para medir o desvio da codificação
    for name, kw in encs.items():
        X = factor_scores(fuzzify_matrix(items, **kw))
        drift = float(np.sqrt(np.mean((X - ref) ** 2)))   # RMS no espaço z-score
        dfg = grid_search(X, m_values=M_RED, n_init=2, random_state=SEED, verbose=False)
        opt = optimal_pair(dfg, primary="XB")
        e = FuzzyCMeansEngine(C_FIXED, m=M_OP, n_init=3, random_state=SEED).fit(X)
        s = cvi_summary(X, e.U_, e.centers_, e.m)
        rows.append({
            "encoding": name, "drift_rms_z": round(drift, 4),
            "c_star": opt["c_star"], "m_star": opt["m_star"],
            "FPC_c6_m2": round(s["FPC"], 3), "XB_c6_m2": round(s["XB"], 2),
            "rho_c6_m2": round(e.boundary_distortion_rate(), 4),
            "rho_c6_m1.5": round(rho_at(X, C_FIXED, 1.5), 4),
        })
        rho_curves[name] = [round(rho_at(X, C_FIXED, m), 4) for m in M_PROFILE]
        print(f"  [{name:10s}] drift={drift:.3f}  (c*,m*)=({opt['c_star']},{opt['m_star']})  "
              f"rho(6,2)={e.boundary_distortion_rate():.3f}")
    return pd.DataFrame(rows), {"m": M_PROFILE, **rho_curves}


# ------------------------------------------------------ (B) reamostragem
def resampling_stability(k_boot=20, n=25_000):
    recs = []
    for b in range(k_boot):
        try:
            items = load_ipip50(ipc_unique=True, sample=n, random_state=1000 + b)
        except FileNotFoundError:
            items, _, _ = make_synthetic_ipip(n=n, c=5, seed=1000 + b)
        X = factor_scores(items)
        dfg = grid_search(X, m_values=M_RED, n_init=1, random_state=SEED, verbose=False)
        opt = optimal_pair(dfg, primary="XB")
        recs.append({
            "boot": b, "c_star": opt["c_star"], "m_star": opt["m_star"],
            "rho_c6_m2": round(rho_at(X, C_FIXED, 2.0), 4),
            "rho_c6_m1.5": round(rho_at(X, C_FIXED, 1.5), 4),
            "FPC_c2_m1.2": round(fpc_at(X, 2, 1.2), 3),
        })
        if (b + 1) % 5 == 0:
            print(f"  bootstrap {b + 1}/{k_boot}")
    df = pd.DataFrame(recs)
    csummary = {
        "k_boot": k_boot,
        "c_star_mode": int(df.c_star.mode().iloc[0]),
        "c_star_counts": {int(k): int(v) for k, v in df.c_star.value_counts().items()},
        "m_star_mean": round(float(df.m_star.mean()), 3),
        "m_star_sd": round(float(df.m_star.std()), 3),
        "rho_c6_m2_mean": round(float(df["rho_c6_m2"].mean()), 4),
        "rho_c6_m2_sd": round(float(df["rho_c6_m2"].std()), 4),
        "rho_c6_m1.5_mean": round(float(df["rho_c6_m1.5"].mean()), 4),
        "rho_c6_m1.5_sd": round(float(df["rho_c6_m1.5"].std()), 4),
    }
    return df, csummary


# ------------------------------------------------------ (C) protótipos
def prototypes(items):
    X = factor_scores(items)                          # z-score OCEAN-5
    e = FuzzyCMeansEngine(C_FIXED, m=M_OP, n_init=10, random_state=SEED).fit(X)
    V = e.centers_                                    # (6, 5) em z-score
    mass = e.U_.sum(axis=1) / e.U_.shape[1]           # "tamanho" difuso de cada protótipo
    order = np.argsort(-mass)
    V, mass = V[order], mass[order]
    rows = []
    for i, (v, w) in enumerate(zip(V, mass), 1):
        rows.append({"prototipo": i, "massa": round(float(w), 3),
                     **{f: round(float(x), 2) for f, x in zip(FACTORS, v)}})
    df = pd.DataFrame(rows)

    # figura: heatmap dos protótipos
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    im = ax.imshow(V, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(FACTORS)
    ax.set_yticks(range(C_FIXED))
    ax.set_yticklabels([f"P{i+1} ({m:.2f})" for i, m in enumerate(mass)])
    for (r, cc), val in np.ndenumerate(V):
        ax.text(cc, r, f"{val:+.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(val) > 0.7 else "black")
    ax.set_title("Protótipos FCM ($c=6$, $m=2$) — escores OCEAN ($z$)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="$z$")
    fig.tight_layout()
    out = FIGDIR / "fig_prototypes.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return df, str(out), float(partition_entropy(e.U_) / np.log(C_FIXED))


def ablation_figure(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for name, ys in curves.items():
        if name == "m":
            continue
        ax.plot(curves["m"], ys, "o-", ms=4, label=name)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xlabel("expoente $m$"); ax.set_ylabel(r"$\rho$ ($c=6$)")
    ax.set_title("Ablação da codificação Likert")
    ax.legend(fontsize=7); ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    out = FIGDIR / "fig_ablation.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return str(out)


def main():
    t0 = time.perf_counter()
    _, meta = load_dataset(space="items", verbose=True)  # só p/ n e proveniência
    try:
        items_raw = load_ipip50(ipc_unique=True, sample=meta["n"], random_state=SEED)
    except FileNotFoundError:
        items_raw, _, _ = make_synthetic_ipip(n=meta["n"], c=5, seed=SEED)

    print("[A] ablação da codificação Likert")
    dfa, curves = ablation(items_raw)
    print(dfa.to_string(index=False), "\n")

    print("[B] estabilidade sob reamostragem (20 bootstraps)")
    dfb, bstat = resampling_stability(k_boot=20, n=meta["n"])
    print(json.dumps(bstat, indent=2), "\n")

    print("[C] protótipos c=6")
    dfp, fig_p, pe_norm = prototypes(items_raw)
    print(dfp.to_string(index=False), f"\n(entropia de partição normalizada = {pe_norm:.3f})\n")

    fig_a = ablation_figure(curves)

    out = _round({
        "ablation": dfa.to_dict(orient="records"),
        "ablation_rho_curves": curves,
        "resampling": {"per_boot": dfb.to_dict(orient="records"), **bstat},
        "prototypes": {"table": dfp.to_dict(orient="records"),
                       "pe_norm": pe_norm, "figure": fig_p},
        "figures": {"ablation": fig_a, "prototypes": fig_p},
        "runtime_seconds": time.perf_counter() - t0,
    })
    (RESULTS / "phase6.json").write_text(json.dumps(out, indent=2))
    dfa.to_csv(RESULTS / "ablation.csv", index=False)
    dfb.to_csv(RESULTS / "resampling.csv", index=False)
    dfp.to_csv(RESULTS / "prototypes.csv", index=False)
    print(f"[fase6] done in {time.perf_counter() - t0:.0f}s -> results/phase6.json")


if __name__ == "__main__":
    main()
