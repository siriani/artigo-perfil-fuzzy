"""Module 2.3 — FCM vs. Hard c-means / Gustafson-Kessel / PCM at (c*, m*).

Reads results/gridsearch_picks.json for (c*, m*) (falls back to c=5, m=2.0), runs the
comparison, writes:
  results/benchmark.csv
  paper/figures/fig_benchmark.png
  paper/figures/fig_membership_hist.png
"""
from __future__ import annotations

import json

from _common import RESULTS, SEED, load_dataset

from fuzzyprofile import FuzzyCMeansEngine
from fuzzyprofile.benchmarks import (
    compare, gustafson_kessel, hard_c_means, possibilistic_c_means,
)


def main():
    X, meta = load_dataset(space="factor")

    try:
        picks = json.loads((RESULTS / "gridsearch_picks.json").read_text())
        c_star = int(picks["euclidean"]["c_star"])
        m_star = float(picks["euclidean"]["m_star"])
    except Exception:
        c_star, m_star = 5, 2.0
    print(f"[bench] (c*, m*) = ({c_star}, {m_star})  n={meta['n']} d={meta['d']}\n")

    df = compare(X, c=c_star, m=m_star, random_state=SEED)
    df.to_csv(RESULTS / "benchmark.csv", index=False)
    print(df.round(4).to_string(index=False))
    print(f"\n[bench] wrote {RESULTS / 'benchmark.csv'}")

    # membership distributions per method (for the histogram figure)
    fcm = FuzzyCMeansEngine(n_clusters=c_star, m=m_star, n_init=5, random_state=SEED).fit(X)
    U_by = {
        "FCM": fcm.U_,
        "Hard c-means": hard_c_means(X, c_star, random_state=SEED)["U"],
        "Gustafson-Kessel": gustafson_kessel(X, c_star, m=m_star, n_init=2, random_state=SEED)["U"],
        "PCM": possibilistic_c_means(X, c_star, m=m_star, random_state=SEED)["U"],
    }

    try:
        import figures
        print("[bench] wrote", figures.benchmark_bars(df))
        print("[bench] wrote", figures.membership_hist(U_by))
    except Exception as e:  # pragma: no cover
        print(f"[bench] figures skipped: {e}")


if __name__ == "__main__":
    main()
