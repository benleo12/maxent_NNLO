"""Export the controlled W fits and their measured or independent references."""

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "archive", type=Path, help="paper_distributions/w/charge_asymmetry.npz"
    )
    p.add_argument("--output", type=Path, default=Path("."))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.archive, allow_pickle=False) as data:
        edges = data["eta_edges"]
        for j, cut in enumerate(data["lepton_pt_min_GeV"]):
            if j == 0:
                reference = np.broadcast_to(
                    data["atlas_born_asymmetry"], (7, len(edges) - 1)
                )
                ref_lowerr, ref_hierr = (
                    data["atlas_total_error_low"],
                    data["atlas_total_error_high"],
                )
            else:
                reference = data["independent_nnlo"][:, j - 1]
                ref_lowerr = ref_hierr = data["independent_nnlo_numerical_sigma"][j - 1]
            columns = {
                "edge": edges,
                "center": np.r_[(edges[1:] + edges[:-1]) / 2, np.nan],
            }
            columns.update(
                ref=reference[0],
                zero=np.zeros_like(reference[0]),
                refminus=ref_lowerr,
                refplus=ref_hierr,
            )
            for name, label in [("R", "basic"), ("R+H+S", "spin")]:
                model = data["models"].tolist().index(name)
                values = data["prediction"][model, :, j]
                difference = values - reference
                for prefix, quantity in [(label, values), (label + "d", difference)]:
                    columns[prefix] = quantity[0]
                    columns[prefix + "lo"] = quantity.min(axis=0)
                    columns[prefix + "hi"] = quantity.max(axis=0)
                columns[label + "err"] = data["prediction_numerical_sigma"][model, j]
            table = [
                values if len(values) == len(edges) else np.r_[values, values[-1]]
                for values in columns.values()
            ]
            np.savetxt(
                args.output / f"w_asymmetry_{cut}.dat",
                np.column_stack(table),
                fmt="%.12g",
                header=" ".join(columns),
                comments="",
            )


if __name__ == "__main__":
    main()
