#!/usr/bin/env python3
r"""JHEP figure: the power-counting rule that places the matching scale.

L(pT; Q) = alpha_s(pT) ln^2(Q/pT) for the three hard scales of this paper.
The criterion L < 0.2 sets the MINIMUM seam (the crossing, dots): 27.9 GeV
for Drell-Yan (Q = m_Z), 27.6 GeV for diphoton (Q = 90 GeV), 37.1 GeV for
gg->H (Q = m_H).  The adopted seams compiled into the fixed-order profiles
are the round numbers at or just above it, 30, 28, 37 (vertical lines);
check_seam() enforces the agreement at every solve.  No data enters.
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
from maxent_upgrade import alpha_s, matching_scale

THR = 0.2
PROCS = [(r"Drell-Yan ($Q=m_Z$)", 91.1876, 30.0, C["maxent"]),
         (r"diphoton ($Q=90$ GeV)", 90.0, 28.0, C["minnlo"]),
         (r"$gg\to H$ ($Q=m_H$)", 125.0, 37.0, C["powheg"])]


def main():
    pt = np.geomspace(5.0, 120.0, 400)
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for lbl, Q, adopted, col in PROCS:
        L = alpha_s(pt) * np.log(Q / pt) ** 2
        xm = matching_scale(Q, thr=THR)
        ax.plot(pt, L, color=col, lw=2.2,
                label=rf"{lbl}: $\geq{xm:.0f}$, seam ${adopted:.0f}$ GeV")
        ax.plot([xm], [THR], "o", color=col, ms=8, zorder=5)
        ax.axvline(adopted, color=col, lw=1.4, ls="--", alpha=0.75)
        print(f"  {lbl:28s} Q={Q:7.2f}  criterion={xm:.2f}  adopted={adopted:.0f} GeV")
    ax.axhline(THR, color="k", lw=1.4, ls="--")
    ax.text(5.4, THR * 1.06, rf"$L={THR}$", fontsize=13)
    ax.set_xscale("log")
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel(r"$L=\alpha_s(p_T)\,\ln^2(Q/p_T)$")
    ax.set_xlim(5, 120); ax.set_ylim(0, 1.6)
    ax.legend(loc="upper right", fontsize=12.5, labelspacing=0.3)
    out = os.path.join(HERE, "fig_seam_rule.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
