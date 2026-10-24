#!/usr/bin/env python3
r"""Build event-level moments from fixed-order events held in Python, then upgrade a prior.

Use this path when your fixed-order calculation gives you weighted phase-space points in
memory (otherwise accumulate the same sums in your own code, see MOMENT_CONSTRUCTION.md).
Toy set-up, no data needed: the "fixed order" is the prior's own events reweighted by a
recoil-tail lift of 40% per 100 GeV, with seven toy scale weights.  The Born towers therefore
carry no resolved change (their SNR stays below 1 and nothing is imposed there), while the
recoil window, rising from 10 to 20 GeV and cut at 200 GeV, is imposed to the depth the
targets resolve.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # if not pip-installed
from maxent_upgrade import upgrade, compute_fo_moments   # noqa: E402

rng = np.random.default_rng(2)
n = 150_000
prior = dict(mll=rng.normal(91.0, 3.0, n).clip(66, 116), y_abs=np.abs(rng.normal(0, 1.1, n)).clip(0, 2.4),
             pT_ll=rng.exponential(8.0, n).clip(0.5, 200), weight=np.ones(n))

lift = 1.0 + 0.4 * prior["pT_ll"] / 100.0                       # the toy higher-order correction
fo = dict(prior, weight=lift, weight_scales=np.array([lift * f for f in (1.0, 1.12, 0.9, 1.08, 0.92, 1.05, 0.96)]))

config = dict(born={"mll": {"range": (66., 116.), "map": "bw"},
                    "y_abs": {"range": (0., 2.4), "map": "lin"}},
              recoil={"pT_ll": {"range": (10., 200.), "map": "log", "soft_lo": 10.,
                                "profile": {"a": 10., "b": 20., "c": 200.}}})
moments = compute_fo_moments(fo, config, n_max=10)
result = upgrade(prior, moments, config)
print(result.describe())
assert (result.weights > 0).all()
