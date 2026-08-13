#!/usr/bin/env python3
r"""How many MIXED moments should Z+jet impose?

Constraining pT_j1 and pT_j2 separately fixes the two marginals; pi - dphi_ll is
determined by their CORRELATION, which only the mixed moments
<T_m(pT_j1) T_n(pT_j2)> carry.  But more moments is not automatically better:
the diphoton case showed that imposing fixed-order moments which are noisy
relative to the information they carry degrades the prediction and costs an
order of magnitude in effective statistics.

So scan the order rather than assume one.  For each NMIX the script reports

  * the fixed-order mixed moments and their errors, so the signal-to-noise of
    what is being imposed is visible;
  * effective statistics after the solve;
  * how well the PREDICTED pi - dphi_ll moments come out against fixed order --
    the only number that decides whether the mixed moments helped.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from maxent_upgrade import upgrade
from nnlojet_moments import (fo_moments_smooth_from_nnlojet, common_seeds,
                             add_profiled_recoil, add_mixed_moments,
                             _moment_over_seeds, _reduce)

ZDIR = "/Users/user/nnlojet-v1.0.2/zj_moments"
RUN, PREFIX = "ZJ_MOMENTS", "ZJ"
CH = os.environ.get("ZJ_CH", "LO,R,V,RR,RV,VV").split(",")
XM, XB, XHI, SOFT = 30.0, 60.0, 1000.0, 10.0
DP_A, DP_B, DP_C = 0.05, 0.20, 4.0
N_DPHI = 6


def prof_w(x, a, b, c):
    lx = np.log(np.maximum(x, 1e-30))
    t = np.clip((lx - np.log(a)) / (np.log(b) - np.log(a)), 0, 1)
    return t * t * t * (t * (t * 6 - 15) + 10) * (x < c)


def cheb_cols(x, lo, hi, n):
    u = 2 * (np.log(np.clip(x, lo, hi)) - np.log(lo)) / (np.log(hi) - np.log(lo)) - 1
    u = np.clip(u, -1, 1)
    out = [np.ones_like(u), u]
    for k in range(2, n + 1):
        out.append(2 * u * out[-1] - out[-2])
    return out


def main():
    seeds = common_seeds(ZDIR, RUN, CH, prefix=PREFIX)
    print(f"channels {'+'.join(CH)};  channel-complete seeds: {len(seeds)}")
    if not seeds:
        sys.exit("no complete seeds")

    z = np.load(os.path.join(HERE, "zj_prior.npz"))
    sel = (np.asarray(z["njet"]) >= 1) & np.isfinite(z["weight"]) & (z["weight"] > 0)
    ev = {k: np.asarray(z[k])[sel] for k in ("mll", "y_abs", "ptj1", "ptj2", "pimdphi")}
    ev["weight"] = np.asarray(z["weight"])[sel]
    print(f"prior events (>=1 jet): {len(ev['weight']):,}")

    # fixed-order target for the PREDICTION
    fo_dphi, er_dphi = [], []
    for n in range(1, N_DPHI + 1):
        m = _moment_over_seeds(ZDIR, RUN, CH, seeds, f"prof_wdphi_{n}", "prof_wdphi_0", PREFIX)
        c, st, sc, tot = _reduce(m, 0)
        fo_dphi.append(c); er_dphi.append(tot)
    fo_dphi = np.array(fo_dphi)
    wd = prof_w(ev["pimdphi"], DP_A, DP_B, DP_C)
    Cd = cheb_cols(ev["pimdphi"], 0.01, np.pi, N_DPHI)

    def dphi_moments(w):
        tot = float(w.sum()); R = float((w * wd).sum() / tot)
        return np.array([float((w * wd * Cd[n]).sum() / tot) / R for n in range(1, N_DPHI + 1)])

    base = dphi_moments(ev["weight"])
    print("\nPREDICTED pi-dphi moments -- |MaxEnt - FO| summed over n=1..6")
    print(f"  prior                                       {np.abs(base - fo_dphi).sum():.4f}")

    for NMIX in (0, 1, 2, 3, 4):
        M = fo_moments_smooth_from_nnlojet(
            ZDIR, RUN, CH, seeds, born_tags={"mll": "mll", "y_abs": "absyz"},
            n_born=6, n_recoil=12, x_match=XM, x_hi=XHI, soft_lo=SOFT,
            recoil_cfg_name="ptj1", norm_born="norm_born",
            w0="prof_wj1_0", wtag="prof_wj1", prefix=PREFIX)
        recoil = {"ptj1": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                           "profile": {"a": XM, "b": XB, "c": XHI}}}
        mixed = {}
        if NMIX:
            sd = common_seeds(ZDIR, RUN, CH,
                              tag=["prof_wj12_0"] + [f"prof_wj12_{m}{n}"
                                                    for m in range(1, NMIX + 1)
                                                    for n in range(1, NMIX + 1)],
                              prefix=PREFIX)
            add_profiled_recoil(M, ZDIR, RUN, CH, sd, "ptj2", wtag="prof_wj2",
                                w0="prof_wj2_0", n_recoil=6, x_match=XM, x_hi=XHI,
                                soft_lo=SOFT, prefix=PREFIX)
            add_mixed_moments(M, ZDIR, RUN, CH, sd, "ptj12", wtag="prof_wj12",
                              w0="prof_wj12_0", n_max=NMIX, prefix=PREFIX)
            recoil["ptj2"] = dict(recoil["ptj1"])
            mixed["ptj12"] = dict(observables=("ptj1", "ptj2"),
                                  range={"ptj1": (SOFT, XHI), "ptj2": (SOFT, XHI)},
                                  map={"ptj1": "log", "ptj2": "log"},
                                  profile={"ptj1": dict(a=XM, b=XB, c=XHI, d=XHI),
                                           "ptj2": dict(a=XM, b=XB, c=XHI, d=XHI)},
                                  n=NMIX)
        cfg = dict(born={"mll": {"range": (66., 116.), "map": "lin"},
                         "y_abs": {"range": (0., 2.4), "map": "lin"}},
                   recoil=recoil, mixed=mixed, followers=["pimdphi"],
                   moment_selection=False)
        try:
            r = upgrade(ev, M, cfg)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  NMIX={NMIX}: FAILED {str(exc)[:52]}")
            continue
        d = np.abs(dphi_moments(r.weights) - fo_dphi).sum()
        tag = "pT_j1 only" if not NMIX else f"+pT_j2 +{NMIX*NMIX} mixed"
        print(f"  NMIX={NMIX} {tag:22s} effN {100*r.effN:5.2f}%  closure {r.closure:.1e}"
              f"   {d:.4f}")


if __name__ == "__main__":
    main()
