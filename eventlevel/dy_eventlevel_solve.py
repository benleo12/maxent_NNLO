#!/usr/bin/env python3
r"""End-to-end DY upgrade driven by EVENT-LEVEL NNLOJET moments.

1. read the NNLOJET profile histograms -> the `moments` dict (nnlojet_moments)
2. load the frozen ATLAS-Z prior events
3. upgrade(events, moments, config)  -- reweight the PS+LO sample to carry the
   fixed-order Born + recoil spectra, using moments that are literal event-level
   weighted sums, not histogram reconstructions.

This is the DY realization of "all moments computed event level".
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from maxent_upgrade import upgrade
from nnlojet_moments import fo_moments_from_nnlojet, seeds_in

NNLO_BASE = os.environ.get("NNLOJET_MOM_DIR",
                           "/Users/user/nnlojet-v1.0.2/dy_profile_poc")
RUN = "DY_MOMENTS"
CHANNELS = ["LO", "R", "V"]        # NLO; extend to the six labels for NNLO
PRIOR = os.path.join(HERE, "dy_prior_atlas_v2.npz")

CONFIG = dict(
    born={"mll":   {"range": (66., 116.), "map": "lin"},
          "y_abs": {"range": (0., 2.4),   "map": "lin"}},
    recoil={"pT_ll": {"range": (0.5, 500.), "map": "log", "soft_lo": 0.5}},
    followers=["phistar", "pT_lead"],
)


def main():
    seeds = seeds_in(NNLO_BASE, RUN)
    print(f"NNLOJET moment dir : {NNLO_BASE}")
    print(f"channels           : {'+'.join(CHANNELS)}   seeds: {seeds}")
    if not seeds:
        sys.exit("no NNLOJET moment .dat files found")

    # ---- (1) event-level fixed-order moments straight from the profiles
    moments = fo_moments_from_nnlojet(
        NNLO_BASE, RUN, CHANNELS, seeds,
        born_tags={"mll": "mll", "y_abs": "absyz"}, recoil_tag="ptz",
        recoil_cfg_name="pT_ll",
        n_born=6, n_recoil=12, x_match=30.0, x_hi=500.0, soft_lo=0.5)

    print("\n---- event-level FO moments (value +/- error) ----")
    for o, d in moments["born"].items():
        cells = " ".join(f"{v:+.4f}({e:.0e})" for v, e in zip(d["values"], d["errors"]))
        print(f"  born  {o:6s}: {cells}")
    rc = moments["recoil"]["pT_ll"]
    cells = " ".join(f"{v:+.4f}({e:.0e})" for v, e in zip(rc["window_values"], rc["window_errors"]))
    print(f"  recoil pT_ll (window [{rc['x_match']:.0f},{rc['x_hi']:.0f}], rate={rc['rate']:.3f}):")
    print(f"    {cells}")

    # ---- (2) prior events
    if not os.path.exists(PRIOR):
        sys.exit(f"prior not found: {PRIOR}")
    P = dict(np.load(PRIOR))
    events = dict(
        mll=P["mll"].astype(float),
        y_abs=np.abs(P["y_ll"]).astype(float),
        pT_ll=P["pT_ll"].astype(float),
        phistar=P["phistar"].astype(float),
        pT_lead=P["pT_lead"].astype(float),
        weight=P["w"].astype(float),
    )
    print(f"\nprior events       : {len(events['weight']):,}")

    # ---- (3) solve
    print("solving upgrade() with event-level moments ...", flush=True)
    res = upgrade(events, moments, CONFIG)

    print("\n================ result (event-level-moment DY upgrade) ================")
    print(f"  effN fraction        : {100*res.effN:.2f}%")
    print(f"  x_match (recoil seam): {res.x_match:.3g} GeV")
    print(f"  chosen #moments      : {res.chosen_moments}")
    print(f"  worst moment closure : {res.closure:.2e}")
    print(f"  weights all positive : {bool((res.weights > 0).all())}")
    if getattr(res, 'band', None) is not None:
        print(f"  band variants        : {len(res.band)}")
    print(f"\n  summary: {res.summary()}")
    np.savez(os.path.join(HERE, "dy_eventlevel_weights.npz"),
             weights=res.weights)
    print("\n  saved weights -> dy_eventlevel_weights.npz")
    return res


if __name__ == "__main__":
    main()
