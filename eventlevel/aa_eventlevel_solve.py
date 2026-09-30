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

GGDIR = os.path.join(os.environ.get("NNLOJET_ROOT", os.path.expanduser("~/nnlojet-v1.0.2")), "gg_moments2")   # clipped-map binary
RUN, PREFIX = "GG_MOMENTS", "GG"
CH = ["LO", "R", "V", "RR", "RV", "VV"]   # full NNLO
# mirrors eval_w_ptaa (pa=28, pb=56, pc=pd=500); see check_seam()
XM, XB, XHI, SOFT = 28.0, 56.0, 500.0, 1.0
Q_HARD = 90.0
MLO, MHI = 80.0, 700.0


# The prior is a set of exclusive slices that concatenate with sigma_i/n_gen,i
# weights.  AA_PRIOR (comma-separated npz list) selects the set; the default is
# the v4 Pythia-internal set of the first submission.  The v5 set adds the
# quark-initiated part from a CKKW-L-merged MG5 aa+aaj sample
# (diphoton_shower_merged.py) and takes the gg box from Pythia alone
# (diphoton_generate_v5.py --channels gg).
AA_PRIOR_SETS = {
    # the 2->2 prior of the first submission (Pythia q qbar -> aa + gg box), exclusive pT-hat slices
    "v4": ("aa_prior_v4_a.npz", "aa_prior_v4_b.npz", "aa_prior_v4_c.npz",
           "aa_prior_v4_d.npz", "aa_prior_v4_e.npz"),
    # the merged prior of the paper: MG5 aa + aaj (CKKW-L in Pythia), leading-photon-pT slices A-D,
    # + the Pythia gg box in pT-hat slices a-e
    "v5": ("aa_prior_v5_mgA.npz", "aa_prior_v5_mgB.npz", "aa_prior_v5_mgC.npz", "aa_prior_v5_mgD.npz",
           "aa_prior_v5_boxa.npz", "aa_prior_v5_boxb.npz", "aa_prior_v5_boxc.npz",
           "aa_prior_v5_boxd.npz", "aa_prior_v5_boxe.npz"),
}
AA_PRIOR_DEFAULT = AA_PRIOR_SETS["v5"]


def prior_files():
    """AA_PRIOR (explicit comma list) > AA_PRIOR_SET (v4 | v5) > the paper's set (v5)."""
    env = os.environ.get("AA_PRIOR")
    if env:
        return [f.strip() for f in env.split(",") if f.strip()]
    return list(AA_PRIOR_SETS[os.environ.get("AA_PRIOR_SET", "v5")])


def load_prior():
    parts = []
    for f in prior_files():
        p = f if os.path.isabs(f) else os.path.join(HERE, f)
        if os.path.exists(p):
            d = dict(np.load(p, allow_pickle=True))
            # EXCLUSIVE pT-hat slices [pthatmin, pthatmax): each event is
            # produced by exactly one slice, so sigma_i/n_gen,i weights simply
            # concatenate.  The v3 set used pTHatMin only, so the slices
            # overlapped and the hard region was counted once per slice --
            # sigma(pt1>200) came out identical from all three.  Sum of the v4
            # slice cross sections is 42.84 pb against the v3 inclusive 42.82.
            # The stored per-event weight is RELATIVE (1 for the Pythia-internal
            # slices; the CKKW-L merging weight, mean 1 over the generated
            # events, for the merged MG5 sample), so each slice is normalised by
            # sigma_i / n_generated_i before concatenation.
            d["weight"] = np.asarray(d["weight"], float) * (float(d["sigma_pb"]) / float(d["n_generated"]))
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
    idx = np.random.default_rng(0).choice(n, min(int(os.environ.get("AA_NMAX", n)), n), replace=False)   # default: ALL fiducial prior events
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
    # A SLICED prior (sigma_i/n_i weights) has an effective count well below N
    # of its own; the slicing-independent statement is the fraction of the
    # prior's Kish count that the tilt retains.
    kish = lambda w: float(np.sum(w) ** 2 / np.sum(np.asarray(w, float) ** 2))
    kp, kq = kish(ev["weight"]), kish(res.weights)
    print(f"  Kish counts: prior {kp:,.0f} ({100*kp/len(ev['weight']):.1f}% of N), "
          f"tilted {kq:,.0f} ({100*kq/len(res.weights):.2f}% of N); retained {100*kq/kp:.1f}% of the prior's")
    out = os.environ.get("AA_WEIGHTS_OUT", "aa_eventlevel_weights.npz")
    np.savez(os.path.join(HERE, out), weights=res.weights, idx=idx,
             prior_files=np.array(prior_files()), effN=res.effN, closure=res.closure)
    print(f"  saved {out}")
    return res


if __name__ == "__main__":
    main()
