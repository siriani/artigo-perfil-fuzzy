"""Figure generation for the paper (matplotlib -> paper/figures/*.png)."""
from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import FIGDIR  # noqa: E402

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})


def pipeline(fname="fig_pipeline.png"):
    """Methodology schematic: Likert -> TFN/IFS fuzzification -> OCEAN-5 -> FCM -> U.

    Not a Mamdani inference system: the front end is a fuzzification step (TFN over the
    1..5 scale), but the core is fuzzy *clustering* (min J_m), not a rule base.
    """
    from matplotlib.patches import FancyArrow, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    W, H, Y0, GAP = 17.0, 22.0, 12.0, 3.6
    xs = [1.0 + i * (W + GAP) for i in range(5)]
    labels = [
        "Respostas\nLikert 1–5",
        None,                                     # box 2: rótulo + faixa TFN (abaixo)
        "Recodificação\n+ projeção\nOCEAN-5",
        "Fuzzy $c$-Means\n$\\min\\,J_m(U,V)$",
        "Pertinência\n$U \\in [0,1]^{\\,c\\times n}$",
    ]
    for i, (x, text) in enumerate(zip(xs, labels)):
        ax.add_patch(FancyBboxPatch(
            (x, Y0), W, H, boxstyle="round,pad=0.25,rounding_size=1.4",
            facecolor="#EEF2F7", edgecolor="#33475b", lw=1.1, clip_on=False))
        if text is not None:
            ax.text(x + W / 2, Y0 + H / 2, text, ha="center", va="center", fontsize=7)
        if i < 4:
            x0 = x + W + 0.3
            ax.add_patch(FancyArrow(x0, Y0 + H / 2, GAP - 0.9, 0, width=0.3,
                                    head_width=1.8, head_length=1.3,
                                    length_includes_head=True, color="#33475b",
                                    clip_on=False))

    # box 2: label on top, TFN membership functions in the lower half
    bx = xs[1]
    ax.text(bx + W / 2, Y0 + H - 3.0, "Fuzzificação\nTFN / IFS",
            ha="center", va="top", fontsize=7)
    fx, fw, fy, fh = bx + 2.4, W - 4.8, Y0 + 2.0, 8.0
    for r in (1, 2, 3, 4, 5):
        a, b, c = max(r - 0.5, 1), r, min(r + 0.5, 5)
        ax.plot(fx + (np.array([a, b, c]) - 1) / 4 * fw,
                fy + np.array([0, 1, 0]) * fh, lw=0.9)

    ax.text(xs[4] + W / 2, Y0 - 4.5,
            r"defuzz. ($\arg\max_i u_{ij}$)  ou  análise de $\rho(c,m)$",
            ha="center", va="center", fontsize=6.5, style="italic")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    out = FIGDIR / fname
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def cost_surfaces(df, indices=("XB", "FPC", "PE"), fname="fig_cost_surface.png"):
    """Heatmaps of validity indices over the (c, m) grid."""
    from fuzzyprofile.gridsearch import cost_surface

    fig, axes = plt.subplots(1, len(indices), figsize=(3.2 * len(indices), 3.0))
    if len(indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        Cm, Mm, Z = cost_surface(df, idx)
        Zp = np.log10(Z) if idx in ("XB", "Kwon") else Z
        im = ax.pcolormesh(Mm, Cm, Zp, shading="auto", cmap="viridis")
        ax.set_xlabel("m"); ax.set_ylabel("c")
        ax.set_title(f"log10({idx})" if idx in ("XB", "Kwon") else idx)
        fig.colorbar(im, ax=ax, fraction=0.046)
        # mark optimum
        best = (np.nanargmin if idx in ("PE", "XB", "FS", "Kwon") else np.nanargmax)(Z)
        bi, bj = np.unravel_index(best, Z.shape)
        ax.plot(Mm[bi, bj], Cm[bi, bj], "r*", ms=12)
    fig.tight_layout()
    out = FIGDIR / fname
    fig.savefig(out); plt.close(fig)
    return out


def convergence(histories, labels, fname="fig_convergence.png"):
    """Normalized objective gap (J_m^k - J_m^inf)/(J_m^1 - J_m^inf) vs iteration."""
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for h, lab in zip(histories, labels):
        h = np.asarray(h, dtype=float)
        gap = (h - h[-1]) / max(h[0] - h[-1], 1e-12)
        gap = np.clip(gap, 1e-6, None)
        ax.plot(range(1, len(h) + 1), gap, marker=".", ms=4, label=f"{lab} ({len(h)} it.)")
    ax.set_xlabel("iteração $k$")
    ax.set_ylabel(r"$(J_m^{(k)}-J_m^{(\infty)})\,/\,(J_m^{(1)}-J_m^{(\infty)})$")
    ax.set_yscale("log"); ax.legend(fontsize=7)
    ax.set_title("Convergência do objetivo (IPIP-50)")
    fig.tight_layout()
    out = FIGDIR / fname
    fig.savefig(out); plt.close(fig)
    return out


def membership_hist(U_by_method, fname="fig_membership_hist.png"):
    """Distribution of the top membership max_i u_ij per method."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for name, U in U_by_method.items():
        top = np.asarray(U).max(axis=0)
        ax.hist(top, bins=40, histtype="step", label=name, density=True)
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel(r"$\max_i u_{ij}$"); ax.set_ylabel("densidade")
    ax.set_title("Pertinência máxima por respondente")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIGDIR / fname
    fig.savefig(out); plt.close(fig)
    return out


def rho_vs_m(m_values, rho, pe_norm, meanmax, c, fname="fig_rho_vs_m.png"):
    """Boundary distortion rate and normalized partition entropy vs. fuzzifier m."""
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(m_values, rho, "o-", label=r"$\rho$ (distorção de fronteira)")
    ax.plot(m_values, pe_norm, "s--", label=r"$V_{PE}/\log c$ (entropia norm.)")
    ax.plot(m_values, meanmax, "^:", label=r"média $\max_i u_{ij}$")
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xlabel("expoente $m$"); ax.set_ylabel("proporção")
    ax.set_title(f"Estrutura da partição vs. $m$  (c = {c}, IPIP-50)")
    ax.legend(fontsize=7); ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    out = FIGDIR / fname
    fig.savefig(out); plt.close(fig)
    return out


def rho_calibration(rp_real, rp_syn, dgmm, fname="fig_rho_calibration.png"):
    """Left: FCM rho(m) real (IPIP-50) vs synthetic 5-cluster control.
    Right: GMM posterior uncertainty vs c (the mixture-model analogue)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.0))
    a1.plot(rp_real["m"], rp_real["rho"], "o-", label="IPIP-50 real (c=6)")
    a1.plot(rp_syn["m"], rp_syn["rho"], "s--", label="sintético 5 clusters (c=5)")
    a1.axhline(0.5, color="k", ls=":", lw=0.8)
    a1.set_xlabel("expoente $m$"); a1.set_ylabel(r"$\rho$ (distorção de fronteira)")
    a1.set_title("FCM: calibração de $\\rho$"); a1.legend(fontsize=7)
    a1.set_ylim(-0.03, 1.03)

    a2.plot(dgmm["c"], dgmm["rho_gmm"], "^-", color="#8172B2",
            label=r"$1-\max_k P(k\,|\,x)$ médio $<0{,}5$")
    a2.plot(dgmm["c"], 1 - dgmm["mean_max_post"], "v--", color="#CCB974",
            label=r"$1-\overline{\max_k P(k\,|\,x)}$")
    a2.set_xlabel("nº de componentes $c$"); a2.set_ylabel("incerteza posterior")
    a2.set_title("GMM (Gerlach): incerteza posterior"); a2.legend(fontsize=7)
    a2.set_ylim(0, 1)
    fig.tight_layout()
    out = FIGDIR / fname
    fig.savefig(out); plt.close(fig)
    return out


def benchmark_bars(df, fname="fig_benchmark.png"):
    """Bar chart: boundary rate + explained variance per method."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.0))
    x = np.arange(len(df))
    a1.bar(x, df["boundary_rate"], color="#4C72B0")
    a1.set_xticks(x); a1.set_xticklabels(df["method"], rotation=30, ha="right")
    a1.set_ylabel(r"taxa de distorção de fronteira $\rho$")
    a2.bar(x, df["explained_variance"], color="#55A868")
    a2.set_xticks(x); a2.set_xticklabels(df["method"], rotation=30, ha="right")
    a2.set_ylabel("variância explicada")
    a2.set_ylim(0, 1)
    fig.tight_layout()
    out = FIGDIR / fname
    fig.savefig(out); plt.close(fig)
    return out
