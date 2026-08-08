#!/usr/bin/env python3
r"""Diphoton upgrade driven by EVENT-LEVEL NNLOJET moments (PRL candidate).

Constrained : m_aa (Born, log [80,700]) and pt_aa (recoil, log [1,500] with the
              smooth profile [11,22]->500, exactly as booked in NNLOJET).
Predicted   : everything else -- |cos theta*|, y_aa, dphi -- which is the point:
              |cos theta*| and y_aa are essentially uncorrelated with the recoil
              we constrained, and ATLAS 1704.03839 measures them.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from maxent_upgrade import upgrade
from nnlojet_moments import fo_moments_smooth_from_nnlojet, common_seeds

GGDIR = "/Users/user/nnlojet-v1.0.2/gg_moments"
RUN, PREFIX = "GG_MOMENTS", "GG"
CH = ["LO", "R", "V"]
XM, XB, XHI, SOFT = 11.0, 22.0, 500.0, 1.0     # profile a,b,c and log-map floor
MLO, MHI = 80.0, 700.0


def load_prior():
    parts = []
    for f in ("aa_prior_s11.npz", "aa_prior_s12.npz"):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            parts.append(dict(np.load(p, allow_pickle=True)))
    if not parts:
        sys.exit("no diphoton prior found")
    g = lambda k: np.concatenate([np.asarray(p[k], float) for p in parts])
    ev = dict(m_aa=g("m_aa"), pt_aa=g("pt_aa"), y_abs=np.abs(g("y_aa")),
              dphi_aa=np.abs(g("dphi_aa")), weight=g("weight"))
    m = np.isfinite(ev["m_aa"]) & (ev["m_aa"] > 0) & np.isfinite(ev["weight"]) & (ev["weight"] > 0)
    return {k: v[m] for k, v in ev.items()}


def main():
    seeds = common_seeds(GGDIR, RUN, CH, prefix=PREFIX)
    print(f"diphoton FO seeds usable: {seeds}  channels {CH}")
    if not seeds:
        sys.exit("no complete diphoton seeds")

    M = fo_moments_smooth_from_nnlojet(
        GGDIR, RUN, CH, seeds,
        born_tags={"m_aa": "maa"},
        n_born=6, n_recoil=12,
        x_match=XM, x_hi=XHI, soft_lo=SOFT,
        recoil_cfg_name="pt_aa",
        norm_born="norm_born", w0="prof_wpt_0", wtag="prof_wpt", prefix=PREFIX)
    b = M["born"]["m_aa"]["values"]; rc = M["recoil"]["pt_aa"]
    print("  <T_n(m_aa)>   :", " ".join(f"{v:+.4f}" for v in b))
    print("  <T_n(pt_aa)>_w:", " ".join(f"{v:+.4f}" for v in rc['window_values'][:6]), "...")
    print(f"  w-rate R = {rc['rate']:.4f}")

    ev = load_prior()
    n = len(ev["weight"])
    idx = np.random.default_rng(0).choice(n, min(1_200_000, n), replace=False)
    ev = {k: v[idx] for k, v in ev.items()}
    print(f"  prior events: {len(ev['weight']):,}")

    cfg = dict(
        born={"m_aa": {"range": (MLO, MHI), "map": "log"}},
        recoil={"pt_aa": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                          "profile": {"a": XM, "b": XB, "c": XHI}}},
        followers=["y_abs", "dphi_aa"],
        moment_selection=False,          # 1 seed -> no spread-based SNR
    )
    print("solving ...", flush=True)
    res = upgrade(ev, M, cfg)
    print(f"\n  effN {100*res.effN:.1f}%   closure {res.closure:.2e}   "
          f"neg-wt {100*np.mean(res.weights<=0):.1f}%   x_match {res.x_match:g}")
    np.savez(os.path.join(HERE, "aa_eventlevel_weights.npz"),
             weights=res.weights, idx=idx)
    print("  saved aa_eventlevel_weights.npz")
    return res


if __name__ == "__main__":
    main()
