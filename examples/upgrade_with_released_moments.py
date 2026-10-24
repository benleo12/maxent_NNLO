#!/usr/bin/env python3
r"""Upgrade YOUR prior with the released fixed-order moments.

    python examples/upgrade_with_released_moments.py dy  my_prior.npz  [out.npz]

`my_prior.npz` holds one array per observable the table constrains (see
`python moments/load_moments.py dy`) and the positive prior weights under 'weight', all of
the same length, for events inside the table's fiducial selection, with the observables
defined as moments/README.md states (Born-level leptons, Collins-Soper angles; Z+jet: events
with at least one jet).  The output holds the same arrays plus 'weight_upgraded', normalized
to the fixed-order fiducial cross section in pb (or to the prior's own sum of weights where
the table's cross section is not validated, as for the Higgs table).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # if not pip-installed
from maxent_upgrade import upgrade, load_moments, sigma_fid   # noqa: E402


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    process, prior = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(prior)[0] + f"_{process}_upgraded.npz"
    moments, config = load_moments(process)
    P = dict(np.load(prior))
    events = {k: np.asarray(v, float) for k, v in P.items() if np.ndim(v) == 1}
    res = upgrade(events, moments, config)          # stops with a clear message if an array is missing
    print(res.describe())
    sig = sigma_fid(process)
    if sig is not None:
        w = res.weights / res.weights.sum() * sig["value"]
        print(f"normalized to the fixed-order fiducial cross section, {sig['value']:.4g} pb")
    else:
        w = res.weights / res.weights.sum() * events["weight"].sum()
        print("this table's cross section is not validated: kept the prior's sum of weights")
    np.savez_compressed(out, **P, weight_upgraded=w)
    print(f"wrote {out} (original arrays + 'weight_upgraded')")


if __name__ == "__main__":
    main()
