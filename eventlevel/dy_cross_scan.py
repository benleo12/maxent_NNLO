#!/usr/bin/env python3
r"""Do the cross moments <T_m(m_ll) T_n(|cos th*|)> predict the lepton pT?

At Born level the lepton transverse momentum is fixed by the mass and the decay
angle,

    pT_l = (m_ll / 2) sin(theta*)   (up to the recoil boost),

so the pair (m_ll, cos theta*) DETERMINES pT_l1 while separate moments of the
two constrain only the marginals.  That makes this the cleanest possible test of
mixed moments: unlike the Z+jet case, where the link from the jet spectra to
pi - dphi was indirect, here the kinematic relation is exact.

The scan sweeps the cross-moment order rather than assuming one, because both
previous cases (diphoton dphi, Z+jet mixed) showed the useful regime is one or
two moments and that imposing noisy high orders costs more than it buys.  For
each order it reports the effective statistics and how well the PREDICTED
pT_l1 moments come out against fixed order, in units of the fixed-order error --
the honest measure when the reference itself is a Monte Carlo estimate.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from maxent_upgrade import upgrade, chebyshev_moment
from nnlojet_moments import (fo_moments_smooth_from_nnlojet, common_seeds,
                             add_mixed_moments, _moment_over_seeds, _reduce)

BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_poc"
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
XM, XHI, SOFT = 30.0, 500.0, 0.5
MLO, MHI = 66.0, 116.0                 # mirrors eval_chebT_mll
PTL1_LO, PTL1_HI = 27.0, 200.0         # mirrors eval_chebT_ptl1
NMOM = 6
PRIOR_FILES = [f"dy_psLO_ext_{i}.npz" for i in (1, 2, 3, 4)]


def load_prior():
    parts = []
    for f in PRIOR_FILES:
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        lp = np.asarray(z["l_plus"], float); lm = np.asarray(z["l_minus"], float)
        mll = np.asarray(z["mll"], float); pT = np.asarray(z["pT_ll"], float)
        yll = np.asarray(z["y_ll"], float); w = np.asarray(z["weight"], float)
        ptp = np.hypot(lp[:, 0], lp[:, 1]); ptm = np.hypot(lm[:, 0], lm[:, 1])
        yp = 0.5 * np.log((lp[:, 3] + lp[:, 2]) / np.maximum(lp[:, 3] - lp[:, 2], 1e-12))
        ym = 0.5 * np.log((lm[:, 3] + lm[:, 2]) / np.maximum(lm[:, 3] - lm[:, 2], 1e-12))
        # |cos theta*| in the Collins-Soper frame, EXACTLY as eval_abscosthstCS_ll
        p1p = lp[:, 3] + lp[:, 2]; p1m = lp[:, 3] - lp[:, 2]
        p2p = lm[:, 3] + lm[:, 2]; p2m = lm[:, 3] - lm[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            cts = np.abs(p1p * p2m - p1m * p2p) / np.maximum(
                mll * np.sqrt(mll ** 2 + pT ** 2), 1e-12)
        m = ((ptp > 27) & (ptm > 27) & (np.abs(yp) < 2.5) & (np.abs(ym) < 2.5)
             & (mll > MLO) & (mll < MHI) & np.isfinite(pT) & np.isfinite(w) & (w > 0))
        parts.append(dict(mll=mll[m], y_abs=np.abs(yll)[m], pT_ll=pT[m],
                          cts=np.clip(np.nan_to_num(cts), 0, 1)[m],
                          pt_l1=np.maximum(ptp, ptm)[m], weight=w[m]))
    if not parts:
        sys.exit("no extended DY prior found")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main():
    need = ["norm_born"] + [f"prof_mct_{m}{n}" for m in range(1, 5) for n in range(1, 5)]
    seeds_x = common_seeds(BASE, "DY_MOMENTS", CH6, tag=need)
    seeds = common_seeds(BASE, "DY_MOMENTS", CH6)
    print(f"seeds: {len(seeds)} total, {len(seeds_x)} with cross moments")
    if not seeds_x:
        sys.exit("cross moments not present yet -- run 56921955 first")

    ev = load_prior()
    n = len(ev["weight"])
    idx = np.random.default_rng(0).choice(n, min(1_500_000, n), replace=False)
    ev = {k: v[idx] for k, v in ev.items()}
    print(f"prior events: {len(ev['weight']):,}")

    # fixed-order target for the PREDICTED pT_l1
    fo, er = [], []
    for k in range(1, NMOM + 1):
        m = _moment_over_seeds(BASE, "DY_MOMENTS", CH6, seeds, f"prof_ptl1_{k}",
                               "norm_born", "Z")
        c, st, sc, tot = _reduce(m, 0); fo.append(c); er.append(tot)
    fo, er = np.array(fo), np.array(er)

    def pull(w):
        mq = chebyshev_moment(ev["pt_l1"], w, NMOM, PTL1_LO, PTL1_HI, "log")
        return np.abs((mq - fo) / np.maximum(er, 1e-12))

    print(f"\nPREDICTED pT_l1: mean |moment - FO| / sigma_FO over n=1..{NMOM}")
    print(f"  prior                                    {pull(ev['weight']).mean():6.2f}")

    for NX in (0, 1, 2, 3, 4):
        M = fo_moments_smooth_from_nnlojet(
            BASE, "DY_MOMENTS", CH6, seeds,
            born_tags={"mll": "mll", "y_abs": "absyz", "cts": "ctl"},
            n_born=NMOM, n_recoil=12, x_match=XM, x_hi=XHI, soft_lo=SOFT)
        born = {"mll": {"range": (MLO, MHI), "map": "bw"},
                "y_abs": {"range": (0., 2.4), "map": "lin"},
                "cts": {"range": (0., 1.), "map": "lin"}}
        mixed = {}
        if NX:
            add_mixed_moments(M, BASE, "DY_MOMENTS", CH6, seeds_x, "mllct",
                              wtag="prof_mct", w0="norm_born", n_max=NX, prefix="Z")
            mixed["mllct"] = dict(observables=("mll", "cts"),
                                  range={"mll": (MLO, MHI), "cts": (0., 1.)},
                                  map={"mll": "bw", "cts": "lin"},
                                  profile=None,     # Born cross moment: no window
                                  n=NX)
        cfg = dict(born=born,
                   recoil={"pT_ll": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                                     "profile": {"a": XM, "b": 2 * XM, "c": XHI}}},
                   mixed=mixed, followers=["pt_l1"], moment_selection=False)
        try:
            r = upgrade(ev, M, cfg)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  NX={NX}: FAILED {str(exc)[:56]}")
            continue
        p = pull(r.weights)
        tag = "no cross" if not NX else f"{NX*NX} cross moments"
        print(f"  NX={NX} {tag:22s} effN {100*r.effN:5.1f}%  closure {r.closure:.1e}"
              f"   {p.mean():6.2f}   (max {p.max():5.2f})")


if __name__ == "__main__":
    main()
