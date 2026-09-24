#!/usr/bin/env python3
r"""The Drell-Yan recoil fixed-order curve on the data edges, drawn EVERYWHERE:
NLO Z+jet (real channels of the NNLO Z run), normalised ABSOLUTELY to the
NNLO fiducial cross section of the same run at the same scale, no window
anchor and no boundary.  Above the seam it is identical, by construction, to
the profile-rate anchor (the n=0 recoil moment is the same on both sides).

Sources: ptz_winfine (30-500 GeV, the paper's 40 seeds) above the seam and,
where available, ptz_full (0-500 GeV, 2 GeV bins, the separate 6-seed
production dy_profile_poc_ptzfull) below it.  Returns per-bin central, scale
half-width (shape-only, each member normalised by its own sigma), and seed
scatter, all as densities normalised to sigma_NNLO; NaN where no histogram
covers a bin.
"""
import os
import numpy as np
from nnlojet_moments import fo_curve, fo_curve_band, common_seeds

BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_log30_hi"
BASE2 = "/Users/user/nnlojet-v1.0.2/dy_profile_poc_ptzfull"
RUN = "DY_MOMENTS"
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
CHR = ("R", "RR", "RV")


def sigma_fid():
    """NNLO fiducial cross section per scale member (mll_fine over the window)."""
    sig = []
    for s_ in range(7):
        lo, hi, d, _ = fo_curve(BASE, RUN, CH6, common_seeds(BASE, RUN, CH6, tag="mll_fine"),
                                "mll_fine", scale_idx=s_)
        inr = (lo >= 66 - 1e-9) & (hi <= 116 + 1e-9)
        sig.append(float(np.sum(d[inr] * (hi - lo)[inr])))
    return np.array(sig)


def fo_ptll(edges, seam=30.0):
    e = np.asarray(edges, float); ctr = 0.5 * (e[:-1] + e[1:]); nb = len(ctr)
    sig = sigma_fid()
    cen = np.full(nb, np.nan); half = np.full(nb, np.nan); stat = np.full(nb, np.nan)
    parts = [(BASE, "ptz_winfine", e[:-1] >= seam - 1e-9)]
    have_full = os.path.isdir(BASE2) and len(common_seeds(BASE2, RUN, CHR, tag="ptz_full")) > 0
    if have_full:
        parts.append((BASE2, "ptz_full", e[1:] <= seam + 1e-9))
    for base, tag, sel in parts:
        fb = fo_curve_band(base, RUN, CHR, common_seeds(base, RUN, CHR, tag=tag), tag,
                           edges=e, members=True)
        if fb is None:
            continue
        _, _, c, _, _, st, mem = fb
        c = c / sig[0]; st = st / sig[0]
        mem_n = np.array([mem[s_] / sig[s_] for s_ in range(7)])
        lo_b = np.minimum(c, np.nanmin(mem_n, 0)); hi_b = np.maximum(c, np.nanmax(mem_n, 0))
        ok = sel & np.isfinite(c)
        cen[ok] = c[ok]; half[ok] = 0.5 * (hi_b - lo_b)[ok]; stat[ok] = st[ok]
    return cen, half, stat, have_full, sig[0]
