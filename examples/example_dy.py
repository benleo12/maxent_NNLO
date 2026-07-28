#!/usr/bin/env python3
r"""End-to-end example: upgrade a PS+LO Drell-Yan sample to PS+NNLO with maxent_upgrade.

Inputs (the frozen ATLAS Z inputs that ship with this project):
  * prior events : dy_prior_atlas_v2.npz   (LO+PS, ~2.79M positive-weight events)
  * FO order N   : nnlo_atlas_run/result/final/nlo.*.dat    (NLO,  7-point scales)
  * FO order N+k : nnlo_atlas_run/result/final/nnlo.*.dat    (NNLO, 7-point scales)  (k=1)

Roles:
  * Born   : m_ll (dilepton invariant mass) and |y_ll| (dilepton rapidity)
  * recoil : pT_ll (dilepton transverse momentum)
  * followers (no FO input): phi*_eta and the leading-lepton pT

Run:
    /Users/user/miniconda3/envs/nnloreweight/bin/python3 maxent_upgrade/example_dy.py
"""
import os
import sys
import time

import numpy as np

# make the package importable when run as a plain script from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maxent_upgrade import upgrade, fo_from_dat

# frozen DY inputs live in the project root (parent of this package); override with env var
BASE = os.environ.get("MAXENT_UPGRADE_DATA", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRIOR = os.path.join(BASE, "dy_prior_atlas_v2.npz")
FO = os.path.join(BASE, "nnlo_atlas_run", "result", "final")


def load(order, files):
    return fo_from_dat([os.path.join(FO, f"{order}.{f}.dat") for f in files])


def main():
    t0 = time.time()
    print("loading prior + FO ...", flush=True)
    P = dict(np.load(PRIOR))
    # events: the public API expects a positive 'weight' array + the observable arrays
    events = dict(
        mll=P["mll"].astype(float),
        y_abs=np.abs(P["y_ll"]).astype(float),
        pT_ll=P["pT_ll"].astype(float),
        phistar=P["phistar"].astype(float),   # follower, no FO input
        pT_lead=P["pT_lead"].astype(float),   # follower, no FO input
        weight=P["w"].astype(float),
    )
    print(f"  prior events: {len(events['weight']):,}", flush=True)

    # the two fixed-order histograms, at the SAME fiducial cuts as the events
    fo_high = {"mll":   load("nnlo", ["mll_a"]),
               "y_abs": load("nnlo", ["absyz_a"]),
               "pT_ll": load("nnlo", ["ptz_fine", "ptz_mid", "ptz_high"])}
    fo_low = {"mll":   load("nlo", ["mll_a"]),
              "y_abs": load("nlo", ["absyz_a"]),
              "pT_ll": load("nlo", ["ptz_fine", "ptz_mid", "ptz_high"])}

    # declare roles + variable maps/ranges; everything else stays at package defaults
    config = dict(
        born={"mll":   {"range": (66., 116.), "map": "lin"},
              "y_abs": {"range": (0., 2.4),   "map": "lin"}},
        recoil={"pT_ll": {"range": (0.5, 500.), "map": "log", "soft_lo": 0.5}},
        followers=["phistar", "pT_lead"],
        # moment_selection=True and snr_threshold=1.0 are the defaults; band=True by default
    )

    print("solving (moment-SNR selection + MaxEnt band) ...", flush=True)
    res = upgrade(events, fo_low, fo_high, config)

    # ---------------------------------------------------------------- report
    print("\n================ moment-SNR spectra (SNR_n = |dmu_n| / sigma_FO) ================")
    for obs, snr in res.moment_snr.items():
        role = "recoil" if obs == "pT_ll" else "born  "
        cells = " ".join(f"{s:4.1f}" for s in snr)
        print(f"  [{role}] {obs:8s} n=1..{len(snr):<2d}: {cells}")
        print(f"           -> resolved up to order {res.chosen_moments[obs]} (SNR>{res.report['moment_selection']['threshold']})")

    print("\n================ upgrade result ================")
    print(f"  effN (effective sample fraction) : {100*res.effN:.2f}%")
    print(f"  x_match (recoil matching scale)  : {res.x_match:.3g} GeV   "
          f"(rule: {res.report['kconv']['rule']})")
    print(f"  chosen #moments per observable   : {res.chosen_moments}")
    print(f"  worst relative moment closure    : {res.closure:.2e}")
    print(f"  all weights strictly positive    : {bool((res.weights > 0).all())}")
    print(f"  negative-weight fraction         : {100*np.mean(res.weights <= 0):.2f}%")
    if res.band is not None:
        print(f"  band variants (scales + rates)   : {len(res.band)}")
    print(f"\n  one-line summary: {res.summary()}")
    print(f"\ndone in {time.time()-t0:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    main()
