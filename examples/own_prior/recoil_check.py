"""Compare fitted Drell–Yan event weights with the released integration results."""
import argparse
from pathlib import Path

import numpy as np


def scales(values):
    return f"{values[0]:.3f} pb  (scale range {values.min():.3f} to {values.max():.3f} pb)"


def compare(reference, paths, observable, column, low):
    edges = reference[f"{observable}_edges"]
    included = edges[:-1] >= low
    lower, upper = edges[:-1][included][0], edges[-1]
    # Integrate each run first to retain correlations between reference bins.
    per_run = reference[f"{observable}_per_run"][:, included, :]
    integrals = (per_run * np.diff(edges)[None, included, None]).sum(axis=1)
    mean = integrals.mean(axis=0)
    error = integrals[:, 0].std(ddof=1) / np.sqrt(len(integrals))
    print(f"\n{lower:g} <= {column} < {upper:g}")
    print(f"  Fixed order: {scales(mean)}, central integration error {error:.3f} pb")
    for path in paths:
        with np.load(path, allow_pickle=False) as events:
            if events["weight_normalization"].item() != "pb":
                raise ValueError(f"{path}: supply an output normalized in pb")
            passed = (events[column] >= lower) & (events[column] < upper)
            prior = events["weight"]
            after = events["weight_upgraded"]
            prior_pb = after.sum() * prior[passed].sum() / prior.sum()
            nominal = after[passed].sum()
            alternatives = events["weight_scale"][:, passed].sum(axis=1)
            print(f"  {path.name}: prior {prior_pb:.3f} pb at the fitted rate, fit {scales(np.r_[nominal, alternatives])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="fit outputs for the DY recoil example")
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1]
                        / "data" / "dy_reference.npz")
    args = parser.parse_args()
    with np.load(args.reference, allow_pickle=False) as reference:
        compare(reference, args.results, "recoil", "pT_ll", reference["recoil_edges"][0])
        compare(reference, args.results, "angle", "abs_costh", 0.9)


if __name__ == "__main__":
    main()
