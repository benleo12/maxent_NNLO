"""A complete uncertainty workflow for user-defined observables, using toy data.

    python examples/custom_uncertainties.py
    python examples/plot_result.py custom_bands.npz mass --label Mass --unit GeV

No released process table is needed. Replace the synthetic targets and their
rates with the corresponding central and varied fixed-order calculations.
These illustrative variations are not estimates of QCD uncertainty.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maxent_upgrade import Variation, compute_fo_moments, upgrade_ensemble


def run(n=4000):
    rng = np.random.default_rng(5)
    mass = rng.uniform(60, 140, n)
    angle = rng.uniform(-1, 1, n)
    events = dict(mass=mass, cos_theta=angle, weight=np.ones(n))
    config = {
        "born": {"mass": {"range": (60, 140), "map": "lin"},
                 "cos_theta": {"range": (-1, 1), "map": "lin"}},
    }
    u = (mass - 100) / 40

    def target(slope):
        sample = dict(events, weight=np.exp(slope * u + 0.2 * angle))
        return compute_fo_moments(sample, config, n_max=4)
    central = target(0.4)
    variations = {"scale": [Variation(target(0.3), cross_section=9.5, label="scale down"),
                            Variation(target(0.5), cross_section=10.8, label="scale up")]}
    result = upgrade_ensemble(events, central, config, variations, cross_section=10,
                              progress=lambda label, r: print(f"{label}: {r.summary()}"))
    hist = result.histogram(mass, np.linspace(60, 140, 21))
    return events, result, hist


if __name__ == "__main__":
    for name in ("custom_bands.npz", "custom_histogram.npz"):
        if Path(name).exists():
            raise SystemExit(f"{name} already exists; run in an empty directory")
    events, result, hist = run()
    np.savez_compressed("custom_bands.npz", **events, **result.arrays())
    np.savez_compressed("custom_histogram.npz", **hist)
    print("Saved custom_bands.npz: events and coherent cross-section weights in pb")
    print("Saved custom_histogram.npz: central density and separate scale band")
