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
OUT = os.path.join(HERE, os.environ.get("ZJ_OUT", "zj_prior.npz"))
JETDEF = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
PTMIN, YMAX = 20.0, 4.4


def main():
    # Sources: parton-level showers that keep the BORN leptons (final lepton +
    # its own QED FSR).  The original build used dy_psLO_partons.npz, a May
    # file with BARE leptons: its mass tower then fought QED radiation, the
    # trap Section 5 of the JHEP describes.  Several files concatenate.
    srcs = os.environ.get("ZJ_SRC", "dy_psLO_born_1.npz").split(",")
    parts = [np.load(os.path.join(HERE, f.strip()), allow_pickle=True) for f in srcs]
    lkp, lkm = (("l_plus_born", "l_minus_born") if all("l_plus_born" in p.files for p in parts)
                else ("l_plus", "l_minus"))
    cat = lambda k, dt=float: np.concatenate([np.asarray(p[k], dt) for p in parts])
    lp, lm = cat(lkp), cat(lkm); w = cat("weight")
    partons = np.concatenate([np.asarray(p["partons"], dtype=object) for p in parts])
    s = lp + lm
    mll = np.sqrt(np.maximum(s[:, 3] ** 2 - (s[:, 0:3] ** 2).sum(1), 0.0))
    yll = 0.5 * np.log((s[:, 3] + s[:, 2]) / np.maximum(s[:, 3] - s[:, 2], 1e-30))
    print(f"sources {srcs}: leptons = {lkp}/{lkm}, {len(mll):,} events, "
          f"mean m_ll in window {np.average(mll[(mll>66)&(mll<116)]):.3f} GeV")
    if os.environ.get("ZJ_FID", "1") == "1":
        # the ATLAS fiducial lepton selection of the DY section, on the same leptons
        pt = lambda v: np.hypot(v[:, 0], v[:, 1])
        eta = lambda v: np.arctanh(np.clip(v[:, 2] / np.maximum(np.sqrt((v[:, :3] ** 2).sum(1)), 1e-30), -1 + 1e-12, 1 - 1e-12))
        keep = ((mll >= 66) & (mll <= 116) & (pt(lp) > 27) & (pt(lm) > 27)
                & (np.abs(eta(lp)) < 2.5) & (np.abs(eta(lm)) < 2.5) & (lp[:, 3] > 0) & (lm[:, 3] > 0))
        print(f"  fiducial lepton selection keeps {keep.sum():,} ({100*keep.mean():.1f}%)")
        lp, lm, w, mll, yll, partons = lp[keep], lm[keep], w[keep], mll[keep], yll[keep], partons[keep]
    n = len(mll)
    lim = int(os.environ.get("ZJ_NMAX", str(n)))
    n = min(n, lim)
    print(f"clustering {n:,} events (anti-kT R=0.4, pT>{PTMIN}, |y|<{YMAX}) ...", flush=True)

    ptj1 = np.zeros(n); ptj2 = np.zeros(n); njet = np.zeros(n, int)
    rap = lambda v: 0.5 * np.log((v[:, 3] + v[:, 2]) / np.maximum(v[:, 3] - v[:, 2], 1e-30))
    yl_p, yl_m = rap(lp), rap(lm)
    fl_p, fl_m = np.arctan2(lp[:, 1], lp[:, 0]), np.arctan2(lm[:, 1], lm[:, 0])
    for i in range(n):
        pl = partons[i]
        if pl is None:
            continue
        pl = np.atleast_2d(np.asarray(pl, float))
        if pl.size == 0 or pl.shape[1] < 4:
            continue
        pj = [fastjet.PseudoJet(float(p[0]), float(p[1]), float(p[2]), float(p[3])) for p in pl]
        jets = fastjet.ClusterSequence(pj, JETDEF).inclusive_jets()
        # the fixed-order run requires min_dr_lj = 0.4: drop jets within
        # Delta R < 0.4 of either lepton (rapidity-azimuth distance)
        lep = ((yl_p[i], fl_p[i]), (yl_m[i], fl_m[i]))
        def far(j):
            for yl, fl in lep:
                dphi = abs(j.phi_std() - fl); dphi = 2 * np.pi - dphi if dphi > np.pi else dphi
                if np.hypot(j.rap() - yl, dphi) < 0.4:
                    return False
            return True
        sel = sorted([j for j in jets if j.pt() > PTMIN and abs(j.rap()) < YMAX and far(j)],
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
