#!/usr/bin/env python3
r"""Systematic improvability: more FO statistics -> more resolved moments -> better
agreement with the fixed order above the matching scale.

For each FO statistics level (number of seeds pooled) we
  1. build the event-level moments and their uncertainties,
  2. measure the moment SNR spectrum  SNR_n = |mu_n(FO) - mu_n(prior)| / sigma_n ,
  3. read off K_max = highest order with SNR > 1  (beyond it we would fit FO noise),
  4. solve with K = 2..12 moments and measure the agreement with the FO shape
     above the matching scale, in the BULK (bins holding 90% of the window rate).

The claim to demonstrate: the accuracy is limited by how many moments the FO
calculation resolves, and that ceiling rises with FO statistics -- i.e. the
method is systematically improvable, not stuck at an intrinsic floor.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from maxent_upgrade import upgrade
from nnlojet_moments import (fo_moments_smooth_from_nnlojet, _load, _sum_channels,
                             _moment_over_seeds, seeds_in, common_seeds)

BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_poc"
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
CH_PT = ["R", "RR", "RV"]          # only these populate pT>0
XM, XHI, SOFT = 30.0, 500.0, 0.5
NSUB = int(os.environ.get("CONV_NSUB", "700000"))


def cheb(u, n):
    u = np.clip(u, -1, 1)
    if n == 0: return np.ones_like(u)
    if n == 1: return u
    t0, t1 = np.ones_like(u), u
    for _ in range(2, n + 1): t0, t1 = t1, 2 * u * t1 - t0
    return t1


def prof_w(x, a, b):
    lx = np.log(np.maximum(x, 1e-30))
    t = np.clip((lx - np.log(a)) / (np.log(b) - np.log(a)), 0, 1)
    return t * t * t * (t * (t * 6 - 15) + 10)


def fo_shape(seeds):
    """FO window density (pT>0 channels, pooled over seeds)."""
    lo = hi = None; tot = None
    for s in seeds:
        for ch in CH_PT:
            r = _load(os.path.join(BASE, f"ch_{ch}", f"Z.DY_MOMENTS.{ch}.ptz_winfine.s{s}.dat"))
            if r is None: continue
            lo, _, hi, v, _ = r
            tot = v[:, 0].copy() if tot is None else tot + v[:, 0]
    return lo, hi, tot


def main():
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v2.npz")))
    n = len(P["w"]); idx = np.random.default_rng(0).choice(n, min(NSUB, n), replace=False)
    ev = dict(mll=P["mll"][idx].astype(float), y_abs=np.abs(P["y_ll"][idx]).astype(float),
              pT_ll=P["pT_ll"][idx].astype(float), weight=P["w"][idx].astype(float))
    x = ev["pT_ll"]; wpr = ev["weight"]

    avail = common_seeds(BASE, "DY_MOMENTS", CH6)   # ALL channels must be present
    print(f"seeds usable (every channel present): {avail}")
    levels = [s for s in (1, 2, 3, 6, len(avail)) if s <= len(avail)]
    levels = sorted(set(levels))

    # prior moments (for the SNR numerator)
    w_ev = prof_w(x, XM, 2 * XM) * (x < XHI)
    u_ev = 2 * (np.log(np.maximum(x, 1e-30)) - np.log(SOFT)) / (np.log(XHI) - np.log(SOFT)) - 1
    den = (wpr * w_ev).sum()
    mu_prior = np.array([(wpr * w_ev * cheb(u_ev, k)).sum() / den for k in range(1, 13)])

    print(f"\n{'seeds':>5} | {'K_max(SNR>1)':>12} | SNR_1..12")
    print("-" * 78)
    snr_by_level = {}
    for ns in levels:
        sd = avail[:ns]
        M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, sd,
                                           born_tags={"mll": "mll", "y_abs": "absyz"},
                                           n_born=6, n_recoil=12,
                                           x_match=XM, x_hi=XHI, soft_lo=SOFT)
        rc = M["recoil"]["pT_ll"]
        v = np.array(rc["window_values"]); e = np.array(rc["window_errors"])
        snr = np.abs(v - mu_prior) / np.maximum(e, 1e-30)
        kmax = max([k for k in range(1, 13) if snr[k-1] > 1], default=0)
        snr_by_level[ns] = (snr, kmax, M)
        print(f"{ns:>5} | {kmax:>12} | " + " ".join(f"{s:5.1f}" for s in snr))

    # agreement vs K, at the largest statistics
    ns = levels[-1]
    _, kmax, M = snr_by_level[ns]
    lo, hi, fv = fo_shape(avail[:ns])
    fe = np.concatenate([lo[:1], hi]); fctr = np.sqrt(fe[:-1] * fe[1:])
    print(f"\nagreement with FO above the matching scale ({ns} seeds pooled)")
    print(f"{'K':>3} | {'bulk median':>11} | {'bulk max':>9} | {'effN':>5} | {'closure':>8}")
    print("-" * 55)
    for K in (2, 4, 6, 8, 10, 12):
        Mk = {"born": M["born"], "recoil": {"pT_ll": dict(M["recoil"]["pT_ll"])}}
        for key in ("window_values", "window_errors", "wprofile_values"):
            if key in Mk["recoil"]["pT_ll"]:
                Mk["recoil"]["pT_ll"][key] = list(Mk["recoil"]["pT_ll"][key])[:K]
        cfg = dict(born={"mll": {"range": (66., 116.), "map": "lin"},
                         "y_abs": {"range": (0., 2.4), "map": "lin"}},
                   recoil={"pT_ll": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                                     "profile": {"a": XM, "b": 2 * XM, "c": XHI}}})
        try:
            res = upgrade(ev, Mk, cfg)
        except Exception as ex:
            print(f"{K:>3} |  solve failed: {str(ex)[:32]}")
            continue
        # FIXED resolution for a clean K-comparison (10 bins across the window)
        nb = int(os.environ.get("CONV_NBINS", "10"))
        ce = np.geomspace(XM, XHI, nb + 1)
        hq, _ = np.histogram(x, ce, weights=res.weights / res.weights.sum()); hq /= np.diff(ce)
        fo_i = np.array([(fv[(fctr >= ce[i]) & (fctr < ce[i+1])]
                          * np.diff(fe)[(fctr >= ce[i]) & (fctr < ce[i+1])]).sum()
                         for i in range(nb)]) / np.diff(ce)
        m = (fo_i > 0) & (hq > 0)
        q = hq[m] / (hq[m] * np.diff(ce)[m]).sum()
        f = fo_i[m] / (fo_i[m] * np.diff(ce)[m]).sum()
        d = np.abs(q / f - 1)
        cum = np.cumsum(f * np.diff(ce)[m]) / (f * np.diff(ce)[m]).sum()
        bulk = cum <= 0.90
        if bulk.sum() == 0: bulk = np.ones_like(d, bool)
        print(f"{K:>3} | {100*np.median(d[bulk]):>10.2f}% | {100*np.max(d[bulk]):>8.2f}% |"
              f" {100*res.effN:>4.0f}% | {res.closure:>8.1e}")


if __name__ == "__main__":
    main()
