"""Fase 6b — validação da taxa de distorção de fronteira rho(c, m).

Rebate o confound "rho sobe só porque c cresce":
  (A) modelo NULO sem estrutura: N(0, Sigma) com Sigma = cov dos escores OCEAN reais.
      Se rho_IPIP(c,m) ~ rho_null(c,m), o IPIP-50 é indistinguível de dados sem tipos.
  (B) bateria sintética: varia c_true, separação, dimensão d e o c_fit imposto.
      Se ha estrutura bem separada, rho fica BAIXO mesmo com c_fit != c_true.

    python experiments/run_rho_validation.py

Escreve results/rho_validation.json + CSVs e paper/figures não (repo é só código;
figuras vão para results/figures/).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from _common import FIGDIR, RESULTS, SEED, load_dataset

from fuzzyprofile import FuzzyCMeansEngine
from fuzzyprofile.data import factor_scores, make_synthetic_ipip

CS = list(range(2, 11))
MS = [1.1, 1.3, 1.5, 1.75, 2.0, 2.5]
N_BIG = 25_000
N_BAT = 6_000


def rho(X, c, m, n_init=2):
    return FuzzyCMeansEngine(c, m=m, n_init=n_init,
                             random_state=SEED).fit(X).boundary_distortion_rate()


def rho_grid(X, label, cs=CS, ms=MS):
    rows = [{"model": label, "c": c, "m": m, "rho": round(rho(X, c, m), 4)}
            for c in cs for m in ms]
    return pd.DataFrame(rows)


# --------------------------------------------------------- (B) bateria sintética
def blobs(n, d, c_true, sep, seed):
    """c_true centros espaçados por ~sep, ruído isotrópico unitário -> (n, d)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, sep, size=(c_true, d))
    y = rng.integers(0, c_true, size=n)
    return centers[y] + rng.normal(0, 1.0, size=(n, d)), y


def battery():
    rows = []
    for c_true in (2, 3, 5, 8):
        for sep in (1.0, 2.0, 3.5):        # sobreposição alta -> baixa
            for d in (2, 5, 10):
                X, _ = blobs(N_BAT, d, c_true, sep, seed=SEED + c_true * 100 + int(sep * 10) + d)
                X = (X - X.mean(0)) / X.std(0)
                for c_fit in CS:
                    for m in (1.5, 2.0):
                        rows.append({
                            "c_true": c_true, "sep": sep, "d": d,
                            "c_fit": c_fit, "m": m,
                            "rho": round(rho(X, c_fit, m, n_init=1), 4),
                        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- figuras
def figures(df_over, df_bat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (1) sobreposição rho vs m (c=6) : IPIP real x nulo x sintético-5
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for lab, sty in [("IPIP-50 real", "o-"), ("nulo N(0,Σ)", "s--"),
                     ("sintético 5 clusters", "^:")]:
        d = df_over[(df_over.model == lab) & (df_over.c == 6)].sort_values("m")
        a1.plot(d.m, d.rho, sty, label=lab)
    a1.axhline(0.5, color="k", ls=":", lw=0.8)
    a1.set_xlabel("expoente $m$"); a1.set_ylabel(r"$\rho$ ($c=6$)")
    a1.set_title("$\\rho$ vs. $m$"); a1.legend(fontsize=7); a1.set_ylim(-0.03, 1.03)

    # (2) rho vs c (m=2) : IPIP real x nulo
    for lab, sty in [("IPIP-50 real", "o-"), ("nulo N(0,Σ)", "s--"),
                     ("sintético 5 clusters", "^:")]:
        d = df_over[(df_over.model == lab) & (df_over.m == 2.0)].sort_values("c")
        a2.plot(d.c, d.rho, sty, label=lab)
    a2.axhline(0.5, color="k", ls=":", lw=0.8)
    a2.set_xlabel("nº de agrupamentos $c$"); a2.set_ylabel(r"$\rho$ ($m=2$)")
    a2.set_title("$\\rho$ vs. $c$"); a2.legend(fontsize=7); a2.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    f1 = FIGDIR / "fig_rho_null.png"; fig.savefig(f1, dpi=150); plt.close(fig)

    # (3) bateria: rho(c_fit) para (c_true x sep), d=5, m=2
    sub = df_bat[(df_bat.d == 5) & (df_bat.m == 2.0)]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    for ax, c_true in zip(axes, (2, 3, 5, 8)):
        for sep in (1.0, 2.0, 3.5):
            d = sub[(sub.c_true == c_true) & (sub.sep == sep)].sort_values("c_fit")
            ax.plot(d.c_fit, d.rho, "o-", ms=3, label=f"sep={sep}")
        ax.axvline(c_true, color="grey", ls=":", lw=1)
        ax.set_title(f"$c_{{true}}={c_true}$"); ax.set_xlabel("$c_{fit}$")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel(r"$\rho$ ($m=2$, $d=5$)"); axes[-1].legend(fontsize=7)
    fig.tight_layout()
    f2 = FIGDIR / "fig_rho_battery.png"; fig.savefig(f2, dpi=150); plt.close(fig)
    return str(f1), str(f2)


def main():
    t0 = time.perf_counter()
    Xf, _meta = load_dataset(space="factor", n=N_BIG)
    Sigma = np.cov(Xf, rowvar=False)
    rng = np.random.default_rng(SEED)
    Xnull = rng.multivariate_normal(np.zeros(Xf.shape[1]), Sigma, size=len(Xf))
    Xnull = (Xnull - Xnull.mean(0)) / Xnull.std(0)
    Xsyn = factor_scores(make_synthetic_ipip(n=len(Xf), c=5, seed=SEED)[0])

    print("[A] rho: IPIP real vs nulo N(0,Σ) vs sintético-5")
    df_over = pd.concat([
        rho_grid(Xf, "IPIP-50 real"),
        rho_grid(Xnull, "nulo N(0,Σ)"),
        rho_grid(Xsyn, "sintético 5 clusters"),
    ], ignore_index=True)
    df_over.to_csv(RESULTS / "rho_null.csv", index=False)

    piv = df_over.pivot_table(index=["c", "m"], columns="model", values="rho")
    piv["delta_IPIP_null"] = piv["IPIP-50 real"] - piv["nulo N(0,Σ)"]
    print(piv.round(3).to_string())
    dmax = float(piv.loc[piv.index.get_level_values("c") >= 3, "delta_IPIP_null"].abs().max())
    print(f"\n[A] |rho_IPIP - rho_null| máx (c>=3) = {dmax:.3f}  "
          f"(pequeno ⇒ IPIP indistinguível de sem-estrutura)\n")

    print("[B] bateria sintética (c_true × sep × d × c_fit)")
    df_bat = battery()
    df_bat.to_csv(RESULTS / "rho_battery.csv", index=False)
    # resumo: com boa separação, rho fica baixo mesmo com c_fit > c_true?
    good = df_bat[(df_bat.sep == 3.5) & (df_bat.m == 2.0) & (df_bat.c_fit > df_bat.c_true)]
    weak = df_bat[(df_bat.sep == 1.0) & (df_bat.m == 2.0) & (df_bat.c_fit > df_bat.c_true)]
    print(f"  sep=3,5, c_fit>c_true, m=2: rho médio = {good.rho.mean():.3f} (máx {good.rho.max():.3f})")
    print(f"  sep=1,0, c_fit>c_true, m=2: rho médio = {weak.rho.mean():.3f} (máx {weak.rho.max():.3f})")

    f1, f2 = figures(df_over, df_bat)
    print("[fig]", f1); print("[fig]", f2)

    out = {
        "rho_overlay": df_over.to_dict(orient="records"),
        "delta_IPIP_null_max_c_ge_3": round(dmax, 4),
        "battery_summary": {
            "well_separated_cfit_gt_ctrue_m2_rho_mean": round(float(good.rho.mean()), 4),
            "well_separated_cfit_gt_ctrue_m2_rho_max": round(float(good.rho.max()), 4),
            "weak_sep_cfit_gt_ctrue_m2_rho_mean": round(float(weak.rho.mean()), 4),
        },
        "battery": df_bat.to_dict(orient="records"),
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    (RESULTS / "rho_validation.json").write_text(json.dumps(out, indent=2))
    print(f"\n[6b] done in {time.perf_counter() - t0:.0f}s -> results/rho_validation.json")


if __name__ == "__main__":
    main()
