#!/usr/bin/env python3
r"""JHEP figure: the per-event weight distributions behind the dilution numbers.

Normalized event weight w/<w> for the MaxEnt tilt and the three matched
generators at the ATLAS Drell-Yan selection.  The (1-2f)^-2 bound counts only
the sign; the measured N/N_eff also pays for the WIDTH of the distribution,
which is why MiNNLO_PS pays 3.67 against its sign-only bound of 3.55, while MC@NLO
(1.26) and POWHEG (1.05), whose weights have fixed magnitude, pay exactly their
bound and the tilt pays 1.03 (2026-10-01 render; numbers printed at run time).

The two outermost bins collect the under- and overflow.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import use_pub_style, C
use_pub_style(base=17)


def dilution(w):
    w = np.asarray(w, float)
    return len(w) * float((w ** 2).sum()) / float(w.sum()) ** 2


def main():
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v3.npz")))
    n = len(P["w"]); idx = np.random.default_rng(0).choice(n, min(1_000_000, n), replace=False)
    wp = P["w"][idx].astype(float)
    BW = dict(np.load(os.path.join(HERE, "dy_band_weights.npz")))
    assert np.array_equal(BW["idx"], idx), "band file subsample mismatch"
    q = np.asarray(BW["central"], float)
    # the tilt factor r = q/p, normalized to unit mean: the weight an analysis
    # sees on top of the (uniform) prior weighting
    r = (q / wp); r = r / r.mean()

    series = [(r"MaxEnt", r, C["maxent"], 2.6)]
    for lbl, f, col in [(r"MiNNLO$_{\mathrm{PS}}$", "dy_minnlo_atlas_v4.npz", C["minnlo"]),
                        (r"MC@NLO", "dy_mcatnlo_atlas_v4.npz", C["mcatnlo"]),
                        (r"POWHEG", "dy_powheg_atlas_v6.npz", C["powheg"])]:
        G = dict(np.load(os.path.join(HERE, f)))
        w = np.asarray(G["w"], float); w = w / w.mean()
        series.append((lbl, w, col, 1.9))

    e = np.linspace(-3.0, 5.0, 161)
    # JHEP print size: the figure is set at 0.66\textwidth, so it is drawn at
    # about 1.5x that width; the legend takes pubstyle's default size (base-4).
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    handles = {}
    for lbl, w, col, lw in series:
        h, _ = np.histogram(np.clip(w, e[0], e[-1]), e, density=True)
        neg = 100.0 * float((w < 0).mean())
        negs = "0" if neg == 0 else f"{neg:.1f}"
        lab = rf"{lbl}: ${dilution(w):.2f}$, ${negs}\%$"
        if lbl == "MaxEnt":
            # the tilt is drawn on top, so the generator spikes near one
            # never hide it (no fill: on a log axis a fill exaggerates the tail)
            handles[lbl] = ax.stairs(h, e, color=col, lw=lw, zorder=5, label=lab)
        else:
            handles[lbl] = ax.stairs(h, e, color=col, lw=lw, zorder=3, label=lab)
        print(f"  {lbl:12s} N/N_eff={dilution(w):6.2f}  neg={neg:5.2f}%")
    ax.axvline(0.0, color="k", lw=1.0, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel(r"event weight $w/\langle w\rangle$")
    ax.set_ylabel(r"probability density")
    ax.set_xlim(e[0], e[-1]); ax.set_ylim(1e-5, 50)
    ax.legend(handles=[handles[k] for k in handles], loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=2, columnspacing=1.2,
              handletextpad=0.5, labelspacing=0.3,
              title=r"$N/N_\mathrm{eff}$, fraction with $w<0$", title_fontsize=13)
    out = os.path.join(HERE, "fig_weight_dists.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
