#!/usr/bin/env python3
r"""Closure test for MIXED two-observable moments.

Separate moments of x and y constrain the two marginals and say nothing about
their joint distribution.  A follower determined by the CORRELATION -- Z+jet's
pi - dphi_ll, given pT_j1 and pT_j2 -- can only be reached by a mixed moment
<T_m(x) T_n(y)>.

This checks the machinery on a case with a known answer: tilt a correlated toy
prior by an exponential that depends on BOTH observables, read the mixed
w-weighted moments off the tilted sample, and confirm the solver recovers them.
It also verifies the thing that makes mixed moments safe -- where either
profile vanishes the feature vanishes, so the prior is untouched there.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # the repository root, if not pip-installed
from maxent_upgrade import upgrade, profile_w, compute_fo_moments
from maxent_upgrade.basis import cheb as _cheb, umap as _umap

LO, HI = 10.0, 1000.0
PA, PB, PC = 30.0, 60.0, 1000.0
NMAX = 3


def mixed_moments(x, y, w, wxy, n):
    """{(m,n): <T_m(x) T_n(y)>_w} and the joint w-rate R, as NNLOJET returns them."""
    Cx = _cheb(_umap(np.clip(x, LO, HI), LO, HI, "log"), n)
    Cy = _cheb(_umap(np.clip(y, LO, HI), LO, HI, "log"), n)
    tot = float(w.sum())
    R = float((w * wxy).sum() / tot)
    out = {}
    for m in range(1, n + 1):
        for k in range(1, n + 1):
            out[(m, k)] = float((w * wxy * Cx[:, m] * Cy[:, k]).sum() / tot) / R
    return out, R


def main():
    rng = np.random.default_rng(11)
    n = 400_000
    # correlated pair: the second jet is softer, and correlated with the first
    pt1 = np.exp(rng.normal(np.log(60.0), 0.8, n)).clip(LO, HI)
    pt2 = (pt1 * np.exp(rng.normal(-0.9, 0.6, n))).clip(LO, HI)
    w0 = np.ones(n)
    ev = dict(ptj1=pt1, ptj2=pt2, weight=w0)

    w1 = profile_w(pt1, PA, PB, PC, PC)
    w2 = profile_w(pt2, PA, PB, PC, PC)
    wxy = w1 * w2

    # a tilt that acts on the CORRELATION, not on either marginal alone
    tilt = np.exp(0.35 * wxy * np.log(pt1 / 80.0) * np.log(pt2 / 40.0))
    wt = w0 * tilt / tilt.mean()

    vals, R = mixed_moments(pt1, pt2, wt, wxy, NMAX)

    M = dict(born={}, recoil={}, mixed={"ptj12": dict(values=vals, rate=R)})
    cfg = dict(born={}, recoil={},
               mixed={"ptj12": dict(observables=("ptj1", "ptj2"),
                                    range={"ptj1": (LO, HI), "ptj2": (LO, HI)},
                                    map={"ptj1": "log", "ptj2": "log"},
                                    profile={"ptj1": dict(a=PA, b=PB, c=PC, d=PC),
                                             "ptj2": dict(a=PA, b=PB, c=PC, d=PC)},
                                    n=NMAX)},
               moment_selection=False)

    res = upgrade(ev, M, cfg)
    q = res.weights
    got, gR = mixed_moments(pt1, pt2, q, wxy, NMAX)

    print(f"solve: effN {100*res.effN:.1f}%  closure {res.closure:.2e}  "
          f"neg-wt {100*np.mean(q <= 0):.1f}%  constraints {res.report['n_constraints']}")
    print(f"{'(m,n)':>7} | {'target':>10} | {'recovered':>10} | {'rel':>9}")
    worst = 0.0
    for k in sorted(vals):
        r = abs(got[k] - vals[k]) / max(abs(vals[k]), 1e-12)
        worst = max(worst, r)
        print(f"{str(k):>7} | {vals[k]:10.5f} | {got[k]:10.5f} | {r:9.2e}")
    rr = abs(gR - R) / R; worst = max(worst, rr)
    print(f"{'rate':>7} | {R:10.5f} | {gR:10.5f} | {rr:9.2e}")

    untouched = wxy == 0
    if untouched.sum() > 100:
        ratio = q[untouched] / w0[untouched]
        spread = float(ratio.std() / ratio.mean())
        print(f"\njoint profile zero on {untouched.sum():,} events: weight-ratio "
              f"spread {spread:.2e} (must be ~0: prior untouched there)")
        worst = max(worst, spread)

    ok = worst < 1e-3
    print(f"\nworst relative miss {worst:.2e} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def test_closure():
    """pytest entry point: the solver recovers a known tilt to better than 1e-3."""
    assert main() == 0


def test_compute_fo_moments_joint():
    """compute_fo_moments builds the joint block (the structure load_moments returns), and the
    joint moments it books on a correlated 'fixed order' are recovered next to the two windowed
    marginal towers."""
    rng = np.random.default_rng(12)
    n = 300_000
    pt1 = np.exp(rng.normal(np.log(60.0), 0.8, n)).clip(LO, HI)
    pt2 = (pt1 * np.exp(rng.normal(-0.9, 0.6, n))).clip(LO, HI)
    ev = dict(ptj1=pt1, ptj2=pt2, weight=np.ones(n))
    prof = dict(a=PA, b=PB, c=PC)
    wxy = profile_w(pt1, PA, PB, PC) * profile_w(pt2, PA, PB, PC)
    lift = np.exp(0.3 * wxy * np.log(pt1 / 80.0) * np.log(pt2 / 40.0))
    fo = dict(ev, weight=lift, weight_scales=np.array([lift * f for f in (1.0, 1.1, 0.9)]))
    cfg = dict(recoil={o: {"range": (LO, HI), "map": "log", "soft_lo": LO, "profile": prof, "transition": "windowed"}
                       for o in ("ptj1", "ptj2")},
               mixed={"ptj12": dict(observables=("ptj1", "ptj2"), range={"ptj1": (LO, HI), "ptj2": (LO, HI)},
                                    map={"ptj1": "log", "ptj2": "log"},
                                    profile={"ptj1": dict(prof, d=PC), "ptj2": dict(prof, d=PC)}, n=NMAX)},
               moment_selection=False)
    mom = compute_fo_moments(fo, cfg, n_max=4)
    blk = mom["mixed"]["ptj12"]
    assert set(blk) >= {"values", "errors", "stat_errors", "scale_errors", "rate", "rate_stat", "rate_scale"}
    assert sorted(blk["values"]) == [(m, k) for m in range(1, NMAX + 1) for k in range(1, NMAX + 1)]
    vals, R = mixed_moments(pt1, pt2, lift, wxy, NMAX)      # the definition, independently
    assert max(abs(blk["values"][k] - vals[k]) for k in vals) < 1e-12 and abs(blk["rate"] - R) < 1e-12
    res = upgrade(ev, mom, cfg)
    got, gR = mixed_moments(pt1, pt2, res.weights, wxy, NMAX)
    worst = max(max(abs(got[k] - vals[k]) / max(abs(vals[k]), 1e-12) for k in vals), abs(gR - R) / R)
    assert worst < 1e-3, worst


if __name__ == "__main__":
    rc = main()
    test_compute_fo_moments_joint(); print("test_compute_fo_moments_joint: PASS")
    sys.exit(rc)
