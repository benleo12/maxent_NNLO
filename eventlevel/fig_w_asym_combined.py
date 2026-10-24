#!/usr/bin/env python3
r"""JHEP figure fig_w_asym.pdf: the W charge asymmetry, both constraint sets in one panel.

The asymmetry A_l(|eta_l|) = (W+ - W-)/(W+ + W-) is a ratio of followers of two
independent upgrades (one per charge, each normalized to its own NNLO fiducial
rate).  This script recomputes nothing: it overlays the arrays that
w_eventlevel.py's asymmetry_figure() saves for each run,

    W_CTW_MODE=off    W_FIG_SUFFIX=_nnlo         -> w_asym_nnlo.npz
        mass, rapidity and recoil towers
    W_CTW_MODE=window W_FIG_SUFFIX=_nnlo_window  -> w_asym_nnlo_window.npz
        the same plus the windowed decay-angle tower (flat for |cos theta*|<0.6,
        C^2 ramp to zero at 0.7)

Drawn: ATLAS 7 TeV (Aad:2011dm, d07-x01-y04), fixed order (NNLO) with its
seven-point band and its Monte Carlo error bars, the LO+PS prior, and the two
MaxEnt samples.  The final sample (with the angle window) carries its scale band
and its bootstrap error bars; the three-tower sample is drawn as a line.
Lower panel: difference to the data.  Set at 0.6\textwidth.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import use_pub_style, C, LS, LW      # noqa: E402
use_pub_style(base=17)
from bandviz import stagger                        # noqa: E402


def main():
    A = dict(np.load(os.path.join(HERE, "w_asym_nnlo.npz")))          # three towers
    B = dict(np.load(os.path.join(HERE, "w_asym_nnlo_window.npz")))   # + angle window
    for k in ("e", "ap", "afo", "afos", "afost", "dy", "deyl"):
        assert np.allclose(A[k], B[k]), f"the two runs disagree on {k}"
    e, xc = B["e"], B["xc"]
    d = dict(x=B["dx"], y=B["dy"], eyl=B["deyl"], exl=B["dexl"])
    ap, afo, afos, afost = B["ap"], B["afo"], B["afos"], B["afost"]
    aq3 = A["aq"]
    aq, aqs, aqb = B["aq"], B["aqs"], B["aqb"]

    fig, (a, r) = plt.subplots(2, 1, figsize=(6.2, 6.6), sharex=True,
                               gridspec_kw={"height_ratios": [2.1, 1.15], "hspace": 0.07})
    # fixed order: grey seven-point band, dotted line, Monte Carlo bars
    a.fill_between(xc, np.minimum(afo, afos.min(0)), np.maximum(afo, afos.max(0)),
                   color=C["band"], alpha=0.55, step="mid", lw=0)
    a.stairs(afo, e, color=C["fo"], ls=LS["fo"], lw=LW["fo"], label=r"fixed order, NNLO")
    a.errorbar(stagger(e, 0, 3), afo, yerr=afost, fmt="none", ecolor=C["fo"], elinewidth=1.0, capsize=1.6)
    a.stairs(ap, e, color=C["prior"], ls="--", lw=2.0, label=r"LO+PS prior")
    a.stairs(aq3, e, color=C["maxent"], ls="-.", lw=1.8, label=r"MaxEnt: mass, rapidity, recoil")
    a.fill_between(xc, np.minimum(aq, aqs.min(0)), np.maximum(aq, aqs.max(0)),
                   color=C["maxent"], alpha=0.20, step="mid", lw=0)
    a.stairs(aq, e, color=C["maxent"], lw=3.0, label=r"MaxEnt: + decay-angle window", zorder=5)
    a.errorbar(stagger(e, 2, 3), aq, yerr=aqb, fmt="none", ecolor=C["maxent"], elinewidth=1.3, capsize=2.2, zorder=5)
    a.errorbar(d["x"], d["y"], yerr=d["eyl"], xerr=d["exl"], fmt="o", color="k", ms=5, lw=1.2,
               capsize=0, label=r"ATLAS", zorder=8)

    r.fill_between(xc, afos.min(0) - d["y"], afos.max(0) - d["y"], color=C["band"], alpha=0.55, step="mid", lw=0)
    r.fill_between(xc, aqs.min(0) - d["y"], aqs.max(0) - d["y"], color=C["maxent"], alpha=0.20, step="mid", lw=0)
    r.stairs(afo - d["y"], e, color=C["fo"], ls=LS["fo"], lw=2.0)
    r.errorbar(stagger(e, 0, 3), afo - d["y"], yerr=afost, fmt="none", ecolor=C["fo"], elinewidth=1.0, capsize=1.6)
    r.stairs(ap - d["y"], e, color=C["prior"], ls="--", lw=1.8)
    r.stairs(aq3 - d["y"], e, color=C["maxent"], ls="-.", lw=1.6)
    r.stairs(aq - d["y"], e, color=C["maxent"], lw=2.4, zorder=5)
    r.errorbar(stagger(e, 2, 3), aq - d["y"], yerr=aqb, fmt="none", ecolor=C["maxent"], elinewidth=1.3,
               capsize=2.2, zorder=5)
    r.errorbar(d["x"], 0 * d["y"], yerr=d["eyl"], fmt="o", color="k", ms=5, lw=1.2, capsize=0, zorder=8)
    r.axhline(0, color="k", lw=0.8)

    a.set_ylabel(r"$A_\ell(|\eta_\ell|)$")
    r.set_ylabel(r"difference to data")
    r.set_xlabel(r"$|\eta_\ell|$")
    a.set_xlim(e[0], e[-1])
    r.set_ylim(-0.085, 0.065)
    # legend order: data, references, then the two samples
    h, l = a.get_legend_handles_labels()
    order = [l.index(x) for x in (r"ATLAS", r"fixed order, NNLO", r"LO+PS prior",
                                  r"MaxEnt: mass, rapidity, recoil", r"MaxEnt: + decay-angle window")]
    a.legend([h[i] for i in order], [l[i] for i in order], loc="upper left",
             labelspacing=0.25, handletextpad=0.5)
    a.text(0.97, 0.05, r"follower", transform=a.transAxes, ha="right", va="bottom", color="0.3")
    out = os.path.join(HERE, "fig_w_asym.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)

    chi = lambda h_, s_=0.0: np.sqrt(np.mean(((h_ - d["y"]) / np.hypot(d["eyl"], s_)) ** 2))
    print(f"  rms pull vs ATLAS: prior {chi(ap):.2f}  MaxEnt(3 towers) {chi(aq3):.2f}  "
          f"MaxEnt(+window) {chi(aq):.2f}  FO {chi(afo):.2f}  FO with its MC error {chi(afo, afost):.2f}")
    print(f"  last-bin pull vs data: prior {(ap[-1]-d['y'][-1])/d['eyl'][-1]:+.2f}  "
          f"MaxEnt(3 towers) {(aq3[-1]-d['y'][-1])/np.hypot(d['eyl'][-1], A['aqb'][-1]):+.2f}  "
          f"MaxEnt(+window) {(aq[-1]-d['y'][-1])/np.hypot(d['eyl'][-1], aqb[-1]):+.2f}  "
          f"FO {(afo[-1]-d['y'][-1])/np.hypot(d['eyl'][-1], afost[-1]):+.2f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
