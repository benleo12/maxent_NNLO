#!/usr/bin/env python3
r"""Build the Z+jet prior: cluster the parton-level DY shower into anti-kT R=0.4
jets and store the observables the ZJ moments constrain.

  pT_j1, pT_j2   : jet transverse momenta (jets pT>20, |y|<4.4, as in the runcard)
  dphi_ll        : azimuthal separation of the two leptons
  mll, y_abs     : Born variables
"""
import os
import sys

import numpy as np
import fastjet

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "zj_prior.npz")
JETDEF = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
PTMIN, YMAX = 20.0, 4.4


def main():
    src = os.path.join(HERE, "dy_psLO_partons.npz")
    z = np.load(src, allow_pickle=True)
    lp = np.asarray(z["l_plus"], float); lm = np.asarray(z["l_minus"], float)
    mll = np.asarray(z["mll"], float); w = np.asarray(z["weight"], float)
    yll = np.asarray(z["y_ll"], float); partons = z["partons"]
    n = len(mll)
    lim = int(os.environ.get("ZJ_NMAX", str(n)))
    n = min(n, lim)
    print(f"clustering {n:,} events (anti-kT R=0.4, pT>{PTMIN}, |y|<{YMAX}) ...", flush=True)

    ptj1 = np.zeros(n); ptj2 = np.zeros(n); njet = np.zeros(n, int)
    for i in range(n):
        pl = partons[i]
        if pl is None:
            continue
        pl = np.atleast_2d(np.asarray(pl, float))
        if pl.size == 0 or pl.shape[1] < 4:
            continue
        pj = [fastjet.PseudoJet(float(p[0]), float(p[1]), float(p[2]), float(p[3])) for p in pl]
        jets = fastjet.ClusterSequence(pj, JETDEF).inclusive_jets()
        sel = sorted([j for j in jets if j.pt() > PTMIN and abs(j.rap()) < YMAX],
                     key=lambda j: -j.pt())
        njet[i] = len(sel)
        if len(sel) > 0: ptj1[i] = sel[0].pt()
        if len(sel) > 1: ptj2[i] = sel[1].pt()
        if (i + 1) % 100000 == 0:
            print(f"  {i+1:,} done", flush=True)

    sl = slice(0, n)
    phi_p = np.arctan2(lp[sl, 1], lp[sl, 0]); phi_m = np.arctan2(lm[sl, 1], lm[sl, 0])
    d = np.abs(phi_p - phi_m); d = np.where(d > np.pi, 2 * np.pi - d, d)
    np.savez_compressed(OUT, mll=mll[sl], y_abs=np.abs(yll[sl]), weight=w[sl],
                        ptj1=ptj1, ptj2=ptj2, njet=njet,
                        dphi_ll=d, pimdphi=np.pi - d)
    print(f"wrote {OUT}")
    print(f"  events with >=1 jet: {(njet>=1).sum():,} ({100*(njet>=1).mean():.1f}%)")
    print(f"  events with >=2 jets: {(njet>=2).sum():,} ({100*(njet>=2).mean():.1f}%)")


if __name__ == "__main__":
    main()
