#!/usr/bin/env python3
r"""Diphoton with EVENT-LEVEL moments vs ATLAS 1704.03839, publication style.

Two constrained observables (m_aa, pT_aa) and three predictions (|cos theta*|,
Delta phi, a_T).  |cos theta*| is essentially uncorrelated with the constrained
recoil, so it is the referee-proof prediction.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import use_pub_style
use_pub_style(base=17)
from aa_vs_data import load_prior_full, dens

D = dict(np.load(os.path.join(HERE, "atlas_aa_8tev.npz"), allow_pickle=True))
W = dict(np.load(os.path.join(HERE, "aa_eventlevel_weights.npz")))

PANELS = [
    ("m_aa",     "m_aa",     r"$m_{\gamma\gamma}$ [GeV]",            True,  "constrained"),
    ("pt_aa",    "pt_aa",    r"$p_T^{\gamma\gamma}$ [GeV]",          True,  "constrained"),
    ("costh_aa", "costh_aa", r"$|\cos\theta^*|$",                    False, "predicted"),
    ("dphi_aa",  "dphi_aa",  r"$\Delta\phi_{\gamma\gamma}$",         False, "predicted"),
    ("at_aa",    "at_aa",    r"$a_T^{\gamma\gamma}$ [GeV]",          True,  "predicted"),
]


def main():
    ev = load_prior_full()
    ev = {k: v[W["idx"]] for k, v in ev.items()}
    wpr, wnw = ev["weight"], W["weights"]

    n = len(PANELS)
    fig, ax = plt.subplots(2, n, figsize=(4.6 * n, 7.4), squeeze=False,
                           gridspec_kw={"height_ratios": [2.1, 1.15], "hspace": 0.07,
                                        "wspace": 0.30})
    for j, (key, dk, lab, logx, role) in enumerate(PANELS):
        a, r = ax[0, j], ax[1, j]
        lo = np.asarray(D[f"{dk}_lo"], float); hi = np.asarray(D[f"{dk}_hi"], float)
        val = np.asarray(D[f"{dk}_val"], float); err = np.asarray(D[f"{dk}_err"], float)
        e = np.concatenate([lo[:1], hi]); ctr = 0.5 * (lo + hi)
        hp, hq = dens(ev[key], wpr, e), dens(ev[key], wnw, e)
        bw = np.diff(e)
        norm = lambda h: h / (h * bw).sum()
        dv = norm(val); pp = norm(hp); qq = norm(hq)
        m = (dv > 0) & (hp > 0)
        med = lambda h: 100 * np.median(np.abs(h[m] / dv[m] - 1))

        a.errorbar(ctr[m], dv[m], yerr=(err / (val * bw).sum())[m], fmt="o", color="k",
                   ms=4, lw=1.1, label=r"ATLAS 1704.03839", zorder=9)
        a.stairs(np.where(m, pp, np.nan), e, color="0.55", ls="--", lw=2.0,
                 label=rf"PS+LO prior ({med(pp):.0f}\%)")
        a.stairs(np.where(m, qq, np.nan), e, color="#d62728", lw=3.0,
                 label=rf"MaxEnt NNLO ({med(qq):.0f}\%)")
        if logx:
            a.set_xscale("log"); r.set_xscale("log")
        a.set_yscale("log"); a.tick_params(labelbottom=False)
        a.set_title(rf"{lab}" + "\n" + rf"\small\textit{{{role}}}", fontsize=15)
        r.axhline(1, color="k", lw=0.8)
        rel = (err / np.maximum(val, 1e-30))
        r.fill_between(ctr[m], (1 - rel)[m], (1 + rel)[m], color="0.75", alpha=0.55, step="mid")
        r.stairs(np.where(m, pp / dv, np.nan), e, color="0.55", ls="--", lw=1.8)
        r.stairs(np.where(m, qq / dv, np.nan), e, color="#d62728", lw=2.4)
        r.set_ylim(0.3, 1.9); r.set_xlabel(lab)
        if j == 0:
            a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}X$")
            r.set_ylabel(r"ratio to data")
        a.legend(loc="lower left", fontsize=12)
    fig.suptitle(r"Diphoton at 8 TeV, event-level NNLO moments: "
                 r"$m_{\gamma\gamma}$ and $p_T^{\gamma\gamma}$ constrained, the rest predicted"
                 "\n" r"\small median $|$ratio$-1|$ vs data in the legend; "
                 r"0\% negative weights", y=1.02)
    out = os.path.join(HERE, "fig_aa_eventlevel.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
