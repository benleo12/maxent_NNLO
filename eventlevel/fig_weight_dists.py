#!/usr/bin/env python3
r"""JHEP figure: the per-event weight distributions behind the dilution numbers.

Normalized event weight w/<w> for the MaxEnt tilt and the three matched
generators at the ATLAS Drell-Yan selection.  The (1-2f)^-2 bound counts only
the sign; the measured N/N_eff also pays for the WIDTH of the distribution,
which is why MiNNLO$_{\\mathrm{PS}}$'s 16.0 far exceeds its sign-only bound of 3.4 while the
tilt sits at 1.09.
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

    series = [(r"MaxEnt tilt", r, C["maxent"], 3.0)]
    for lbl, f, col in [(r"MiNNLO$_{\mathrm{PS}}$", "dy_minnlo_atlas_v4.npz", C["minnlo"]),
                        (r"MC@NLO", "dy_mcatnlo_atlas_v4.npz", C["mcatnlo"]),
                        (r"POWHEG", "dy_powheg_atlas_v6.npz", C["powheg"])]:
        G = dict(np.load(os.path.join(HERE, f)))
        w = np.asarray(G["w"], float); w = w / w.mean()
        series.append((lbl, w, col, 1.9))

    e = np.linspace(-3.0, 5.0, 161)
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for lbl, w, col, lw in series:
        h, _ = np.histogram(np.clip(w, e[0], e[-1]), e, density=True)
        neg = 100.0 * float((w < 0).mean())
        ax.stairs(h, e, color=col, lw=lw,
                  label=rf"{lbl} ($N/N_\mathrm{{eff}}={dilution(w):.2f}$, "
                        rf"${neg:.0f}\%\,w<0$)")
        print(f"  {lbl:12s} N/N_eff={dilution(w):6.2f}  neg={neg:5.2f}%")
    ax.axvline(0.0, color="k", lw=1.0, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel(r"event weight $w/\langle w\rangle$")
    ax.set_ylabel(r"probability density")
    ax.set_xlim(e[0], e[-1]); ax.set_ylim(1e-5, 50)
    ax.legend(loc="upper right", fontsize=12, labelspacing=0.3)
    out = os.path.join(HERE, "fig_weight_dists.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
