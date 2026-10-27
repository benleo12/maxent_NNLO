#!/usr/bin/env python3
r"""The uncertainty bands, from the released per-run tables.

    python examples/bands_with_released_moments.py dy my_prior.npz [n_boot] [out.npz]

Loops over the same call (README, "Uncertainty bands"):
  * scale:     one re-solve per scale choice k = 1..6 of the seven-point variation
               (load_moments(process, scale=k));
  * target statistics: one re-solve per bootstrap replica of the independent fixed-order
               runs (load_moments(process, runs=resample));
  * matching ("ggh" and "aa" only): one re-solve per window variant A, B, C of the recoil
               tower (load_moments(process, window=V)); the envelope of the nominal solve and
               the three variants is the matching uncertainty;
  * sample statistics: the usual per-bin error of the weighted events, sum of q_i^2.
Every variant imposes the central solve's constraint set (fixed_orders) and starts from its
multipliers (lam0).  The output holds the central weights, the six scale variants, the
bootstrap replicas and, where released, the window variants; the per-bin envelope of the
scale variants is the band, the bootstrap spread (+) the sample's own error the error bar.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # if not pip-installed
from maxent_upgrade import upgrade, load_moments, released   # noqa: E402


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    process, prior = sys.argv[1], sys.argv[2]
    n_boot = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    out = sys.argv[4] if len(sys.argv) > 4 else os.path.splitext(prior)[0] + f"_{process}_bands.npz"
    events = {k: np.asarray(v, float) for k, v in np.load(prior).items() if np.ndim(v) == 1}

    moments, config = load_moments(process)
    central = upgrade(events, moments, config)
    frozen = {**config, "fixed_orders": dict(central.report["moment_selection"]["chosen"]),
              "lam0": central.report["lam"]}
    print(f"central: {central.summary()}")

    scale_w = []
    for k in range(1, 7):
        m_k, _ = load_moments(process, scale=k, check_seam=False)
        r = upgrade(events, m_k, frozen); scale_w.append(r.weights)
        print(f"  scale {k} ({released.table(process)['scales'][k]}): effN {100*r.effN:.1f}%")

    n_runs = released.table(process)["n_independent_runs"]
    rng = np.random.default_rng(1); boot_w = []
    for b in range(n_boot):
        m_b, _ = load_moments(process, runs=rng.integers(0, n_runs, n_runs), check_seam=False)
        boot_w.append(upgrade(events, m_b, frozen).weights)
    print(f"  {n_boot} bootstrap replicas of the {n_runs} runs: effN "
          f"{100*min(1/(len(q)*(q**2).sum()) for q in boot_w):.1f}-{100*max(1/(len(q)*(q**2).sum()) for q in boot_w):.1f}%")

    window_w = {}
    for V in released.windows(process):
        if V == "nominal":
            continue
        m_v, c_v = load_moments(process, window=V)
        r = upgrade(events, m_v, {**c_v, "fixed_orders": frozen["fixed_orders"], "lam0": frozen["lam0"]})
        window_w[f"window_{V}"] = r.weights
        print(f"  window {V} {c_v['recoil'][next(iter(c_v['recoil']))]['profile']}: effN {100*r.effN:.1f}%")

    np.savez_compressed(out, central=central.weights, scale_w=np.array(scale_w), boot_w=np.array(boot_w), **window_w)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
