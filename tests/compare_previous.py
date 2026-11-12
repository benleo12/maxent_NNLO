"""Compare reweighting results with an earlier checkout in a separate directory.

    python tests/compare_previous.py /path/to/earlier/checkout

Runs on a compute host or CI. Both implementations receive the same synthetic
inputs. The comparison covers Born, blended and windowed recoil constraints,
joint moments, soft constraints and warm starts.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np


PROBE = r'''
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from maxent_upgrade import compute_fo_moments, upgrade

rng = np.random.default_rng(23)
n = int(sys.argv[3])
events = dict(mass=rng.uniform(60, 140, n),
              x=np.exp(rng.uniform(np.log(1), np.log(800), n)),
              y=np.exp(rng.uniform(np.log(1), np.log(800), n)),
              weight=rng.uniform(0.5, 1.5, n))
fixed_order = dict(events, weight=events['weight'] * np.exp(
    0.15 * (events['mass'] - 100) / 40 + 0.1 * np.log(events['x'])))
born = {'mass': {'range': (60., 140.), 'map': 'lin'}}
profile = {'a': 20., 'b': 40., 'c': 500.}
recoil = {o: {'range': (20., 500.), 'soft_lo': 20., 'map': 'log', 'profile': profile}
          for o in ('x', 'y')}
joint = {'xy': {'observables': ('x', 'y'), 'range': {o: (20., 500.) for o in ('x', 'y')},
                'map': {'x': 'log', 'y': 'log'}, 'profile': {'x': profile, 'y': profile}, 'n': 2}}
windowed = {o: {**c, 'transition': 'windowed'} for o, c in recoil.items()}
cases = [dict(born=born), dict(born=born, recoil={'x': recoil['x']}),
         dict(born=born, recoil=recoil), dict(born=born, recoil=windowed, mixed=joint)]
saved = {}
for i, config in enumerate(cases):
    moments = compute_fo_moments(fixed_order, config, n_max=4)
    for soft in (False, True):
        cfg = {**config, 'soft_constraints': soft}
        result = upgrade(events, moments, cfg)
        key = f'{i}_{soft}'
        saved[key] = result.weights
        saved[key + '_targets'] = result.report['mu']
        saved[key + '_stats'] = np.array([result.effN, result.effN_retained, result.max_pull])
        warm = upgrade(events, moments, {**cfg, 'fixed_orders': result.chosen_moments,
                                         'lam0': result.report['lam']})
        saved[key + '_warm'] = warm.weights
np.savez(sys.argv[2], **saved)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("--events", type=int, default=12000)
    args = parser.parse_args()
    baseline = Path(args.baseline).resolve()
    current = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory() as temporary:
        for name, checkout in (("old", baseline), ("new", current)):
            subprocess.run([sys.executable, "-c", PROBE, str(checkout), f"{temporary}/{name}.npz", str(args.events)],
                           cwd=temporary, env=env, check=True)
        with np.load(f"{temporary}/old.npz") as old, np.load(f"{temporary}/new.npz") as new:
            assert old.files == new.files
            for key in old.files:
                np.testing.assert_allclose(new[key], old[key], rtol=1e-12, atol=1e-14, err_msg=key)
            print(
                f"PASS: {len(old.files)} weight, target and diagnostic arrays agree with the previous implementation")


if __name__ == "__main__":
    main()
