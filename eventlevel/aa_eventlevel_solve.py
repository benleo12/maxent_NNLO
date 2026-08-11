#!/usr/bin/env python3
r"""Diphoton upgrade driven by EVENT-LEVEL NNLOJET moments (PRL candidate).

Constrained : the BORN observables m_aa (log [80,700]) and |cos theta*| (lin
              [0,1]), plus the RECOIL observable pt_aa (log [1,500] with the
              smooth profile [28,56]->500, exactly as booked in NNLOJET).
Predicted   : y_aa, dphi_aa, a_T -- measured by ATLAS 1704.03839 and never fit.

Which slot an observable belongs in is not a free choice.  A Born observable is
defined at Born level and infrared-safe over its whole range, so its fixed-order
moments may be imposed everywhere.  A recoil observable vanishes at Born level
and its fixed-order prediction diverges in the soft limit, so it may only be
constrained through the profile, above the matching scale.  Putting pi - dphi in
the Born slot makes the dual infeasible -- see aa_angular_test.py.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from maxent_upgrade import upgrade, check_seam
from nnlojet_moments import fo_moments_smooth_from_nnlojet, common_seeds

GGDIR = "/Users/user/nnlojet-v1.0.2/gg_moments"
RUN, PREFIX = "GG_MOMENTS", "GG"
CH = ["LO", "R", "V", "RR", "RV", "VV"]   # full NNLO
# mirrors eval_w_ptaa (pa=28, pb=56, pc=pd=500); see check_seam()
XM, XB, XHI, SOFT = 28.0, 56.0, 500.0, 1.0
Q_HARD = 90.0
MLO, MHI = 80.0, 700.0


def load_prior():
    parts = []
    for f in ("aa_prior_v3_s1a.npz", "aa_prior_v3_s1b.npz", "aa_prior_v3_s1c.npz"):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            d = dict(np.load(p, allow_pickle=True))
            # pT-hat SLICES: the stored per-event weight is 1, so each slice must be
            # normalised by sigma_i / n_generated_i before they can be concatenated.
            n = len(np.asarray(d["weight"], float))
            d["weight"] = np.full(n, float(d["sigma_pb"]) / float(d["n_generated"]))
            parts.append(d)
    if not parts:
        sys.exit("no diphoton prior found")
    g = lambda k: np.concatenate([np.asarray(p[k], float) for p in parts])
    g1, g2 = g("g1"), g("g2")
    m_aa, pt_aa = g("m_aa"), g("pt_aa")
    p1p = (g1[:, 3] + g1[:, 2]); p1m = (g1[:, 3] - g1[:, 2])
    p2p = (g2[:, 3] + g2[:, 2]); p2m = (g2[:, 3] - g2[:, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        cts = np.abs(p1p * p2m - p1m * p2p) / np.maximum(
            m_aa * np.sqrt(m_aa**2 + pt_aa**2), 1e-12)
    dphi = np.abs(g("dphi_aa"))
    ev = dict(m_aa=m_aa, pt_aa=pt_aa, y_abs=np.abs(g("y_aa")),
              costh_aa=np.clip(np.nan_to_num(cts), 0, 1),
              dphi_log=np.maximum(np.pi - dphi, 1e-8),
              dphi_aa=dphi, weight=g("weight"))
    m = np.isfinite(ev["m_aa"]) & (ev["m_aa"] > 0) & np.isfinite(ev["weight"]) & (ev["weight"] > 0)
    return {k: v[m] for k, v in ev.items()}


def main():
    check_seam(XM, Q_HARD, label="diphoton")
    born_tags = {"m_aa": "maa", "costh_aa": "cts", "dphi_log": "dpa"}
    # select seeds on EVERY tag that is summed, not just the Born denominator
    need = ["norm_born", "prof_wpt_0"] + [f"prof_{t}_1" for t in born_tags.values()] \
           + [f"prof_wpt_{n}" for n in range(1, 13)]
    seeds = common_seeds(GGDIR, RUN, CH, tag=need, prefix=PREFIX)
    print(f"diphoton FO seeds usable: {len(seeds)}  channels {CH}")
    if not seeds:
        sys.exit("no complete diphoton seeds")

    M = fo_moments_smooth_from_nnlojet(
        GGDIR, RUN, CH, seeds,
        born_tags=born_tags,
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
        born={"m_aa": {"range": (MLO, MHI), "map": "log"},
              # |cos theta*| is a genuine BORN observable (defined at Born level,
              # infrared-safe over its whole range).  Constraining it pins the
              # m-cos correlation that the photon pT cuts impose and that mass
              # moments alone cannot reach: it takes cos theta* from 17.8% to
              # 6.3% against ATLAS and simultaneously relieves m_aa 12.0 -> 9.5%.
              "costh_aa": {"range": (0.0, 1.0), "map": "lin"}},
        # pi - dphi is deliberately NOT a Born constraint.  It is a RECOIL
        # observable: identically zero at Born level and Sudakov-divergent as
        # dphi -> pi, so imposing it over its full range contradicts the recoil
        # profile (which preserves the shower below the seam) and the dual has
        # no positive-weight solution at any moment order (N=3,4,6 all fail).
        # Left as a prediction it still improves, 31.8% -> 8.4% vs ATLAS.
        recoil={"pt_aa": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                          "profile": {"a": XM, "b": XB, "c": XHI}}},
        followers=["y_abs"],
        moment_selection=False,
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
