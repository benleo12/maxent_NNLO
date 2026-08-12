#!/usr/bin/env python3
r"""Closure test for TWO profiled recoil observables in one solve.

The diphoton upgrade needs to constrain pT_gg AND pi-dphi_gg simultaneously,
because constraining pT alone leaves the shower's pT<->dphi correlation wrong.
This checks the multi-recoil path on a case where the answer is known exactly:
tilt the prior by a known exponential in both observables, read the resulting
w-weighted moments off the tilted sample, and confirm the solver recovers them.

If this passes, a failure on real data is physics, not plumbing.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from maxent_upgrade import upgrade, profile_w, chebyshev_moment

XM_PT, XB_PT, XHI_PT, SOFT_PT = 28.0, 56.0, 500.0, 1.0
XM_DP, XB_DP, XHI_DP, SOFT_DP = 0.3, 0.6, np.pi, 0.01
N_REC = 6


def wmoments(x, w, wprof, a, b, mp, n):
    """(<T_n>_w, R) exactly as NNLOJET's profile histograms return them."""
    tot = float(w.sum())
    R = float((w * wprof).sum() / tot)
    mu = chebyshev_moment(x, w * wprof, n, a, b, mp)
    return mu, R


def main():
    rng = np.random.default_rng(7)
    n = 400_000
    # a toy prior with a realistic pT spectrum and a pT<->dphi correlation
    pt = np.exp(rng.normal(np.log(12.0), 1.1, n)).clip(SOFT_PT, XHI_PT)
    dphi_r = np.pi - np.clip(pt / 45.0 * np.exp(rng.normal(0, 0.35, n)), 1e-6, np.pi - 1e-6)
    pimd = np.pi - dphi_r
    w0 = np.ones(n)

    ev = dict(pt_aa=pt, dphi_log=pimd, weight=w0)

    # ---- build a TARGET by a known tilt in both observables -----------------
    wp_pt = profile_w(pt, XM_PT, XB_PT, XHI_PT, XHI_PT)
    wp_dp = profile_w(pimd, XM_DP, XB_DP, None, None)
    tilt = np.exp(0.45 * wp_pt * np.log(pt / 30.0) - 0.30 * wp_dp * np.log(pimd / 0.5))
    wt = w0 * tilt / tilt.mean()

    mu_pt, R_pt = wmoments(pt, wt, wp_pt, SOFT_PT, XHI_PT, "log", N_REC)
    mu_dp, R_dp = wmoments(pimd, wt, wp_dp, SOFT_DP, XHI_DP, "log", N_REC)

    M = dict(born={}, recoil={
        "pt_aa": dict(x_match=XM_PT, x_hi=XHI_PT, soft_lo=SOFT_PT,
                      window_values=mu_pt, wprofile_values=mu_pt,
                      rate=R_pt, wprofile_rate=R_pt),
        "dphi_log": dict(x_match=XM_DP, x_hi=XHI_DP, soft_lo=SOFT_DP,
                         window_values=mu_dp, wprofile_values=mu_dp,
                         rate=R_dp, wprofile_rate=R_dp)})
    cfg = dict(born={},
               recoil={"pt_aa": {"range": (SOFT_PT, XHI_PT), "map": "log",
                                 "soft_lo": SOFT_PT,
                                 "profile": {"a": XM_PT, "b": XB_PT, "c": XHI_PT}},
                       "dphi_log": {"range": (SOFT_DP, XHI_DP), "map": "log",
                                    "soft_lo": SOFT_DP,
                                    "profile": {"a": XM_DP, "b": XB_DP}}},
               moment_selection=False)

    res = upgrade(ev, M, cfg)
    q = res.weights

    got_pt, gR_pt = wmoments(pt, q, wp_pt, SOFT_PT, XHI_PT, "log", N_REC)
    got_dp, gR_dp = wmoments(pimd, q, wp_dp, SOFT_DP, XHI_DP, "log", N_REC)

    print(f"solve: effN {100*res.effN:.1f}%  closure {res.closure:.2e}  "
          f"neg-wt {100*np.mean(q <= 0):.1f}%  constraints {res.report['n_constraints']}")
    print(f"{'n':>3} | {'pT target':>10} {'pT got':>10} {'rel':>9} | "
          f"{'dphi target':>11} {'dphi got':>10} {'rel':>9}")
    worst = 0.0
    for i in range(N_REC):
        r1 = abs(got_pt[i] - mu_pt[i]) / max(abs(mu_pt[i]), 1e-12)
        r2 = abs(got_dp[i] - mu_dp[i]) / max(abs(mu_dp[i]), 1e-12)
        worst = max(worst, r1, r2)
        print(f"{i+1:>3} | {mu_pt[i]:10.5f} {got_pt[i]:10.5f} {r1:9.2e} | "
              f"{mu_dp[i]:11.5f} {got_dp[i]:10.5f} {r2:9.2e}")
    rr1 = abs(gR_pt - R_pt) / R_pt; rr2 = abs(gR_dp - R_dp) / R_dp
    worst = max(worst, rr1, rr2)
    print(f"rate | {R_pt:10.5f} {gR_pt:10.5f} {rr1:9.2e} | "
          f"{R_dp:11.5f} {gR_dp:10.5f} {rr2:9.2e}")

    # the shower must be untouched where BOTH profiles vanish
    untouched = (wp_pt == 0) & (wp_dp == 0)
    if untouched.sum() > 100:
        r = q[untouched] / w0[untouched]
        spread = float(r.std() / r.mean())
        print(f"\nboth profiles zero on {untouched.sum():,} events: "
              f"weight-ratio spread {spread:.2e} (must be ~0: shape preserved)")
        worst = max(worst, spread)

    ok = worst < 1e-3
    print(f"\nworst relative miss {worst:.2e} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
