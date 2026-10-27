#!/usr/bin/env python3
r"""The released diphoton moments imposed on a TOY prior (runs in seconds, no data needed).

The constraint set is the default one: the Born towers of the diphoton mass (log map) and of
|cos theta*| (Collins-Soper, linear map), and the recoil pT_gg inside the window that rises
from 28 to 56 GeV and is cut at 500 GeV.  The toy prior only has to cover the same phase
space; replace it with your own LO+PS diphoton events (ATLAS 8 TeV fiducial selection, see
`python moments/load_moments.py aa`) to do the real upgrade.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # if not pip-installed
from maxent_upgrade import upgrade, load_moments   # noqa: E402

rng = np.random.default_rng(7)
n = 400_000
events = dict(
    m_aa=(80.0 * np.exp(rng.exponential(0.45, n))).clip(80.0, 700.0),   # falling continuum above 80 GeV
    abs_costh_aa=rng.uniform(0.0, 1.0, n),                                  # |cos theta*|
    pt_aa=np.exp(rng.normal(np.log(15.0), 1.0, n)).clip(1.0, 500.0),     # recoil: Sudakov-like peak, long tail
    weight=np.ones(n),
)
moments, config = load_moments("aa")
result = upgrade(events, moments, config)
print(result.describe())
assert (result.weights > 0).all()
