"""A new process with two user-defined observables and a correlation.

    python examples/custom_observables.py
    python examples/plot_result.py custom_result.npz mass --label Mass --unit GeV

This is synthetic data, not a QCD prediction. For a real process, replace prior
and fixed_order with shower and fixed-order arrays, using identical cuts and
observable definitions. No process registration or solver changes are needed.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maxent_upgrade import compute_fo_moments, upgrade


def run(n_events=4000):
    rng = np.random.default_rng(8)
    mass = rng.uniform(60.0, 140.0, n_events)
    angle = rng.uniform(-1.0, 1.0, n_events)
    prior = dict(mass=mass, cos_theta=angle, weight=np.ones(n_events))

    # Known correction: a mass shape, an angular shape and a correlation.
    u = (mass - 100.0) / 40.0
    target_weights = np.exp(0.4 * u + 0.25 * (2 * angle**2 - 1) + 0.6 * u * angle)
    fixed_order = dict(prior, weight=target_weights)
    config = {
        "born": {
            "mass": {"range": (60.0, 140.0), "map": "lin"},
            "cos_theta": {"range": (-1.0, 1.0), "map": "lin"},
        },
        "mixed": {
            "mass_angle": {
                "observables": ("mass", "cos_theta"),
                "range": {"mass": (60.0, 140.0), "cos_theta": (-1.0, 1.0)},
                "map": {"mass": "lin", "cos_theta": "lin"},
                "n": 2,
            },
        },
    }
    moments = compute_fo_moments(fixed_order, config, n_max=4)
    separate = upgrade(prior, moments, {**config, "mixed": {}})
    joint = upgrade(prior, moments, config)
    print(joint.describe())
    correlation = u * angle
    target = target_weights / target_weights.sum()
    print("\nJoint moment <u(mass) cos(theta)>:")
    for name, weights in (("Target", target), ("Separate moments", separate.weights), ("Joint moments", joint.weights)):
        print(f"  {name:18} {weights @ correlation:.6f}")
    return dict(prior, weight_upgraded=joint.weights,
                weight_separate=separate.weights, weight_target=target)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=int, default=4000)
    parser.add_argument("--output", default="custom_result.npz")
    args = parser.parse_args()
    if args.events < 100:
        parser.error("use at least 100 events for this demonstration")
    if Path(args.output).exists():
        parser.error("output already exists; choose another --output path")
    np.savez_compressed(args.output, **run(args.events))
    print(f"\nSaved {args.output}. All demonstration weights are normalized to unit area.")


if __name__ == "__main__":
    main()
