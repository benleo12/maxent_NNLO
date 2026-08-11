#!/usr/bin/env python3
r"""Does the method meet its OWN specification?

  below the matching scale : reweighted SHAPE == parton-shower shape?
  above the matching scale : reweighted     == fixed order?

The first is tested as reweighted/prior with the (physical) constant 1/Z divided
out -- a flat ratio means the shower shape is untouched.
The second is tested as reweighted/FO bin-by-bin inside the window.

Reports max and median |ratio-1| in each region, and the moment closure for
reference (closure is a moment-space statement; this is the distribution-space
one, which is what a per-mille claim would have to rest on).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from maxent_upgrade import upgrade
from nnlojet_moments import fo_moments_smooth_from_nnlojet, _load

BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_poc"
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
SEEDS = None
XM, XHI, SOFT = 30.0, 500.0, 0.5


def fo_curve(tag, channels, seeds):
    """channel-summed, seed-averaged FO density (central scale)."""
    lo = hi = None; acc = []
    for s in seeds:
        c = None
        for ch in channels:
            r = _load(os.path.join(BASE, f"ch_{ch}", f"Z.DY_MOMENTS.{ch}.{tag}.s{s}.dat"))
            if r is None:
                continue
            lo_i, _, hi_i, v, _ = r
            if c is None:
                c = v[:, 0].copy(); lo, hi = lo_i, hi_i
            elif v.shape[0] == c.shape[0]:
                c = c + v[:, 0]
        if c is not None:
            acc.append(c)
    if not acc:
        return None
    return lo, hi, np.mean(acc, axis=0)


def main():
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v2.npz")))
    n = len(P["w"]); idx = np.random.default_rng(0).choice(n, min(1_500_000, n), replace=False)
    ev = dict(mll=P["mll"][idx].astype(float), y_abs=np.abs(P["y_ll"][idx]).astype(float),
              pT_ll=P["pT_ll"][idx].astype(float), weight=P["w"][idx].astype(float))
    M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, (SEEDS or common_seeds(BASE, 'DY_MOMENTS', CH6)),
                                       born_tags={"mll": "mll", "y_abs": "absyz"},
                                       n_born=6, n_recoil=12, x_match=XM, x_hi=XHI, soft_lo=SOFT)
    cfg = dict(born={"mll": {"range": (66., 116.), "map": "lin"},
                     "y_abs": {"range": (0., 2.4), "map": "lin"}},
               recoil={"pT_ll": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                                 "profile": {"a": XM, "b": 2 * XM, "c": XHI}}})
    res = upgrade(ev, M, cfg)
    print(f"moment closure (moment space)      : {res.closure:.2e}")
    print(f"effN {100*res.effN:.0f}%   neg-wt {100*np.mean(res.weights<=0):.0f}%\n")

    x = ev["pT_ll"]; wpr = ev["weight"]; wpo = res.weights

    # ---------- below the seam: shower shape preserved? ----------
    e = np.geomspace(1.0, XM, 16)
    ctr = np.sqrt(e[:-1] * e[1:]); bw = np.diff(e)
    hp, _ = np.histogram(x, e, weights=wpr / wpr.sum()); hp /= bw
    hq, _ = np.histogram(x, e, weights=wpo / wpo.sum()); hq /= bw
    ok = hp > 0
    r = hq[ok] / hp[ok]
    z = r.mean()
    dev = np.abs(r / z - 1)
    print("BELOW the matching scale  (reweighted shape vs shower shape)")
    print(f"  constant rescale 1/Z      = {z:.4f}  (physical: window carries the FO rate)")
    print(f"  shape deviation  median   = {100*np.median(dev):.2f} %")
    print(f"  shape deviation  max      = {100*np.max(dev):.2f} %")

    # ---------- above the seam: FO reproduced? ----------
    # only channels with pT>0 populate the window (Born channels are identically 0
    # and would otherwise donate their [10,500] bin EDGES to the sum)
    fc = fo_curve("ptz_winfine", ["R", "RR", "RV"], SEEDS)
    if fc is None:
        print("\nABOVE: no FO reference found"); return
    lo, hi, fv = fc
    fe = np.concatenate([lo[:1], hi])
    fctr = np.sqrt(fe[:-1] * fe[1:])
    sel = (fctr >= XM) & (fctr < XHI) & (fv > 0)
    hq2, _ = np.histogram(x, fe, weights=wpo / wpo.sum()); hq2 /= np.diff(fe)
    # normalise both to unit integral over the window, so we compare SHAPES
    wsel = np.diff(fe)[sel]
    q = hq2[sel] / (hq2[sel] * wsel).sum()
    f = fv[sel] / (fv[sel] * wsel).sum()
    d = np.abs(q / f - 1)
    print("\nABOVE the matching scale  (reweighted vs fixed order, window shape)")
    print(f"  bins compared             = {sel.sum()}")
    print(f"  |ratio-1| median          = {100*np.median(d):.2f} %")
    print(f"  |ratio-1| max             = {100*np.max(d):.2f} %")
    # BULK = bins holding the central 90% of the window cross-section
    cum = np.cumsum(f * wsel) / (f * wsel).sum()
    bulk = cum <= 0.90
    print(f"  BULK (90% of window rate, {bulk.sum()} bins):")
    print(f"     median {100*np.median(d[bulk]):.2f} %   max {100*np.max(d[bulk]):.2f} %")
    print(f"  TAIL (last 10% of rate): median {100*np.median(d[~bulk]):.2f} %"
          f"   max {100*np.max(d[~bulk]):.2f} %")
    # coarser bins = the resolution the moments actually control
    for nb in (12, 8, 6):
        ce = np.geomspace(XM, XHI, nb + 1)
        hqc, _ = np.histogram(x, ce, weights=wpo / wpo.sum()); hqc /= np.diff(ce)
        # rebin FO onto the coarse grid by integrating the fine curve
        fo_int = np.array([ (fv[(fctr >= ce[i]) & (fctr < ce[i+1])]
                             * np.diff(fe)[(fctr >= ce[i]) & (fctr < ce[i+1])]).sum()
                            for i in range(nb) ]) / np.diff(ce)
        m = (fo_int > 0) & (hqc > 0)
        qq = hqc[m] / (hqc[m]*np.diff(ce)[m]).sum(); ff = fo_int[m] / (fo_int[m]*np.diff(ce)[m]).sum()
        print(f"  at {nb:2d} bins: median {100*np.median(np.abs(qq/ff-1)):5.2f} %   "
              f"max {100*np.max(np.abs(qq/ff-1)):5.2f} %")


if __name__ == "__main__":
    main()
