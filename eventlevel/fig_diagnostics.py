#!/usr/bin/env python3
r"""Two method diagnostics for the long paper.

LEFT  -- where to put the matching scale.  D(x) = |FO/shower - 1| as a function
         of pT: fixed order and the shower disagree badly at small pT (the FO
         spectrum diverges) and again far in the tail (the shower runs out of
         hard radiation).  The seam is placed where they agree, i.e. at the
         minimum of D.  This replaces the old two-bound `fig_criterion`.

RIGHT -- what the reweighting cannot do.  The tilt q_i = p_i e^{lambda.Phi}/Z has
         support(q) = support(p): where the prior has no events, no choice of
         lambda creates any.  Shown for the diphoton Delta phi, where the prior
         holds 31 events below Delta phi = 1 and both prior and reweighted sit
         far below the data.  This replaces `fig_support_diag`.
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

FO = os.path.join(HERE, "nnlo_atlas_run", "result", "final")
XM = 30.0


def read_dat(path):
    lo, hi, val = [], [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        lo.append(float(p[0])); hi.append(float(p[2])); val.append(float(p[3]))
    return np.array(lo), np.array(hi), np.array(val)


def fo_spectrum(order):
    lo, hi, val = [], [], []
    for tag in ("ptz_fine", "ptz_mid", "ptz_high"):
        p = os.path.join(FO, f"{order}.{tag}.dat")
        if not os.path.exists(p):
            continue
        a, b, c = read_dat(p)
        lo.append(a); hi.append(b); val.append(c)
    if not lo:
        return None
    lo = np.concatenate(lo); hi = np.concatenate(hi); val = np.concatenate(val)
    o = np.argsort(lo)
    return lo[o], hi[o], val[o]


def main():
    fig, ax = plt.subplots(1, 2, figsize=(13.6, 5.6))

    # ---------------- LEFT: the seam ----------------
    # Standard power counting: fixed order is untrustworthy where the
    # resummation logarithms are large, alpha_s(pT) ln^2(Q/pT) ~ 1.  Data-blind,
    # no tuning, one prescription for every process through the hard scale Q.
    # (This retires the old kconv / minimal-discrepancy rules, which walked down
    #  to 8-10 GeV where the product is ~0.9, i.e. deep in the Sudakov region.)
    a = ax[0]

    def alphas(mu, mz=91.1876, az=0.118, nf=5):
        b0 = (33 - 2*nf)/(12*np.pi); b1 = (153 - 19*nf)/(24*np.pi**2)
        L = np.log(np.maximum(mu, 1.0)**2/mz**2); al = az
        for _ in range(60):
            al = az/(1 + az*b0*L + az**2*(b1/b0)*np.log(np.maximum(1 + az*b0*L, 1e-9)))
        return al

    THR = 0.2
    pt = np.geomspace(3.0, 70.0, 500)   # matching region; ln(Q/pT)->0 above
    # Processes are distinguished by LINE STYLE, not colour: the colours in this
    # scheme mean generators everywhere else, and reusing them here for
    # processes would make the same colour mean two different things.
    for Q, lab, dsh in [(91.1876, r"Drell--Yan, $Q=m_Z$", (None, None)),
                        (125.0,   r"$gg\to H$, $Q=m_H$",  (6, 2)),
                        (90.0,    r"diphoton, $Q\simeq m_{\gamma\gamma}$", (2, 2))]:
        v = alphas(pt)*np.log(Q/np.minimum(pt, 0.99*Q))**2
        ln, = a.plot(pt, v, lw=2.4, color="0.15", label=lab)
        if dsh[0]: ln.set_dashes(dsh)
        ok = np.where(v < THR)[0]
        if len(ok):
            a.plot([pt[ok[0]]], [THR], "o", color=C["maxent"], ms=9, zorder=5)
            print(f"    {lab}: x_match = {pt[ok[0]]:.1f} GeV")
    a.axhline(THR, color=C["maxent"], lw=2.0, ls="--",
              label=rf"threshold {THR}")
    a.axhline(1.0, color="0.4", lw=1.2, ls=":")
    a.set_ylim(0.02, 4.0)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"$p_T$ [GeV]")
    a.set_ylabel(r"$\alpha_s(p_T)\,\ln^2(Q/p_T)$")
    a.set_title(r"Where to match: resummation power counting", fontsize=16); a.legend(fontsize=11.5, loc="upper right")

    # ---------------- RIGHT: support ----------------
    b = ax[1]
    parts = [dict(np.load(os.path.join(HERE, f), allow_pickle=True))
             for f in ("aa_prior_v4_a.npz", "aa_prior_v4_b.npz", "aa_prior_v4_c.npz",
                       "aa_prior_v4_d.npz", "aa_prior_v4_e.npz")
             if os.path.exists(os.path.join(HERE, f))]
    g = lambda k: np.concatenate([np.asarray(p[k], float) for p in parts])
    dphi, wa = np.abs(g("dphi_aa")), g("weight")
    e2 = np.linspace(0, np.pi, 33)
    cnt, _ = np.histogram(dphi, e2)
    ctr2 = 0.5 * (e2[:-1] + e2[1:])
    b.stairs(np.maximum(cnt, 0.3), e2, color="0.15", lw=2.4, fill=False)
    b.set_yscale("log")
    b.axhspan(0.3, 100, color=C["maxent"], alpha=0.10)
    b.axhline(100, color=C["maxent"], lw=1.6, ls=":")
    n_low = int((dphi < 1).sum())
    b.text(0.12, 3.0, rf"only ${n_low}$ events below $\Delta\phi=1$",
           color=C["maxent"], fontsize=14)
    b.set_xlabel(r"$\Delta\phi_{\gamma\gamma}$")
    b.set_ylabel(r"prior events per bin")
    b.set_title(r"What reweighting cannot do: prior support", fontsize=16)
    print(f"  gamma-gamma: {int((dphi<1).sum())} prior events with dphi<1, "
          f"weight fraction {100*wa[dphi<1].sum()/wa.sum():.4f}%")

    out = os.path.join(HERE, "fig_diagnostics.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
