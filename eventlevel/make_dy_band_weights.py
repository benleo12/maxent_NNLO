#!/usr/bin/env python3
r"""Build the Drell-Yan uncertainty-band caches consumed by the DY figures:
  dy_band_weights.npz      -- ATLAS-side pipeline (fig_dy_spectra, fig_dy_eventlevel,
                              phistar_prediction, fig_weight_dists): same event
                              order and config as fig_dy_eventlevel.main
  dy_band_weights_ext.npz  -- moments figure (fig_nnlo_born): its own loader
Each cache holds central weights, 6 scale re-solves (7-point variations of the
targets, warm-started from the central multipliers) and N_BOOT seed-bootstrap
re-solves.  The figures verify `central` against their own solve and refuse
the cache if it is stale, so this must be re-run whenever the prior changes.
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from maxent_upgrade import upgrade
from nnlojet_moments import fo_moments_smooth_from_nnlojet, common_seeds
BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_poc"; CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
XM, XHI, SOFT = 30.0, 500.0, 0.5; N_BOOT = 20
KW = dict(born_tags={"mll": "mll", "y_abs": "absyz"}, n_born=6, n_recoil=12,
          x_match=XM, x_hi=XHI, soft_lo=SOFT)
CFG = dict(born={"mll": {"range": (66., 116.), "map": "bw"},
                 "y_abs": {"range": (0., 2.4), "map": "lin"}},
           recoil={"pT_ll": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                             "profile": {"a": XM, "b": 2 * XM, "c": XHI}}})

def ev_atlas():
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v3.npz")))
    n = len(P["w"]); idx = np.random.default_rng(0).choice(n, min(1_000_000, n), replace=False)
    return dict(mll=P["mll"][idx].astype(float), y_abs=np.abs(P["y_ll"][idx]).astype(float),
                pT_ll=P["pT_ll"][idx].astype(float), phistar=P["phistar"][idx].astype(float),
                weight=P["w"][idx].astype(float), _idx=idx), {**CFG, "followers": ["phistar"]}

def ev_ext():
    import fig_nnlo_born as F
    ev = F.load_prior(); n = len(ev["weight"])
    idx = np.random.default_rng(0).choice(n, min(1_500_000, n), replace=False)
    return {k: v[idx] for k, v in ev.items()}, {**CFG, "followers": ["pt_l1", "pt_l2"]}

def build(ev, cfg, out):
    idx = ev.pop("_idx", None)
    seeds = common_seeds(BASE, "DY_MOMENTS", CH6)
    M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, seeds, **KW)
    res = upgrade(ev, M, cfg); lam0 = res.report["lam"]
    # Freeze the central solve's SNR-chosen moment counts for every variant:
    # a variant whose own veto drops or adds a moment would change the feature
    # count and the warm start (and the band) would no longer be like-for-like.
    chosen = dict(res.report["moment_selection"]["chosen"])
    cfg = {**cfg, "moment_selection": False,
           "born_N": {o: int(chosen[o]) for o in cfg["born"]},
           "recoil_N": int(chosen["pT_ll"])}
    print(f"{out}: N={len(ev['weight']):,} effN {100*res.effN:.1f}% closure {res.closure:.1e}  "
          f"chosen {chosen}", flush=True)
    scale_w = []
    for s in range(1, 7):
        Ms = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, seeds, scale_idx=s, **KW)
        r = upgrade(ev, Ms, {**cfg, "lam0": lam0}); scale_w.append(r.weights)
        print(f"   scale {s}: effN {100*r.effN:.1f}%", flush=True)
    rng = np.random.default_rng(1); boot_w = []
    for b in range(N_BOOT):
        sb = list(rng.choice(seeds, size=len(seeds), replace=True))
        Mb = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, sb, **KW)
        boot_w.append(upgrade(ev, Mb, {**cfg, "lam0": lam0}).weights)
    extra = {"idx": idx} if idx is not None else {}
    np.savez_compressed(os.path.join(HERE, out), central=res.weights,
                        scale_w=np.array(scale_w), boot_w=np.array(boot_w), **extra)
    print(f"   wrote {out} ({len(scale_w)} scale + {len(boot_w)} boot)", flush=True)

if __name__ == "__main__":
    which = sys.argv[1:] or ["atlas", "ext"]
    build(*ev_atlas(), "dy_band_weights.npz")   # one prior, one cache, shared by every DY figure
