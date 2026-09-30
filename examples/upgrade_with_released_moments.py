#!/usr/bin/env python3
r"""Upgrade YOUR prior with the released fixed-order moments.

    python examples/upgrade_with_released_moments.py dy  my_prior.npz  [out_weights.npz]

The .npz must hold one array per observable the table names (see
`python moments/load_moments.py dy`) and a positive 'weight' array, all of the
same length, for events inside the table's fiducial region and with the
observables defined as the table states (Born-level leptons, the Collins-Soper
angle, etc.; MOMENT_CONSTRUCTION.md gives the definitions).  The output is the
same events with the new positive weights, normalized to the fixed-order
fiducial cross section in pb.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from maxent_upgrade import upgrade            # noqa: E402
from moments.load_moments import load, table   # noqa: E402


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    process, prior = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(prior)[0] + f"_{process}_upgraded.npz"
    moments, config = load(process)
    need = list(config["born"]) + list(config["recoil"]) + [o for m in config.get("mixed", {}).values() for o in m["observables"]]
    P = dict(np.load(prior))
    missing = [k for k in dict.fromkeys(need + ["weight"]) if k not in P]
    if missing:
        sys.exit(f"prior lacks the arrays {missing}; it needs {sorted(set(need))} and 'weight'")
    events = {k: np.asarray(P[k], float) for k in dict.fromkeys(need + ["weight"])}
    res = upgrade(events, moments, config)
    sig = table(process)["sigma_fid_pb"]["value"]
    w = res.weights / res.weights.sum() * sig
    print(f"{process}: {len(w):,} events, effN {100*res.effN:.1f}%, closure {res.closure:.1e}, "
          f"negative weights {int((w <= 0).sum())}, moments imposed {dict(res.chosen_moments)}; sum of weights = {sig:.4g} pb")
    np.savez_compressed(out, **{k: P[k] for k in P}, weight_upgraded=w)
    print(f"wrote {out} (original arrays + 'weight_upgraded')")


if __name__ == "__main__":
    main()
