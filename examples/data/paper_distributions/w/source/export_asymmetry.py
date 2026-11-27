"""Export the fitted lepton charge asymmetries and their fixed-order and ATLAS ratios."""
import argparse
import json
from pathlib import Path

import numpy as np


def add_scales(columns, name, members):
    columns[name] = members[0]
    columns[name + "lo"] = members.min(axis=0)
    columns[name + "hi"] = members.max(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="paper_distributions/w/charge_asymmetry.npz")
    parser.add_argument("--baseline", type=Path, help="w_joint/charge_asymmetry.json")
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.baseline or args.archive.resolve().parents[2] / "w_joint/charge_asymmetry.json"
    baseline = json.loads(source.read_text())["reference"]["eta20"]
    prior = np.genfromtxt(args.archive.with_name("lo_ps_prior.csv"), delimiter=",", names=True)

    with np.load(args.archive, allow_pickle=False) as data:
        edges = data["eta_edges"]
        baseline_ref = np.asarray(baseline["coherent_scales"])
        measurements = data["atlas_born_asymmetry"]
        for j, cut in enumerate(data["lepton_pt_min_GeV"]):
            if cut == 20:
                fixed_order = baseline_ref
                integration_error = np.asarray(baseline["stat"])
            else:
                fixed_order = data["independent_nnlo"][:, j - 1]
                integration_error = data["independent_nnlo_numerical_sigma"][j - 1]
            if not np.all(np.isfinite(fixed_order) & (fixed_order > 0)):
                raise ValueError(f"Missing fixed-order denominator for {cut} GeV")
            columns = dict(edge=edges, center=np.r_[(edges[1:] + edges[:-1]) / 2, np.nan],
                           reference=fixed_order[0], referr=integration_error,
                           unit=np.ones_like(measurements), ferr=integration_error / fixed_order[0])
            unweighted = prior[prior["cut_GeV"] == cut]
            np.testing.assert_array_equal(unweighted["eta_low"], edges[:-1])
            np.testing.assert_array_equal(unweighted["eta_high"], edges[1:])
            columns["prior"] = unweighted["asymmetry"]
            columns["priorfo"] = unweighted["asymmetry"] / fixed_order[0]
            if cut == 20:
                columns["priordata"] = unweighted["asymmetry"] / measurements
                columns.update(data=measurements, dlo=data["atlas_total_error_low"],
                               dhi=data["atlas_total_error_high"],
                               dexplo=1 - data["atlas_total_error_low"] / measurements,
                               dexphi=1 + data["atlas_total_error_high"] / measurements)
            for name, short in [("R", "basic"), ("R+H+S", "spin")]:
                model = data["models"].tolist().index(name)
                values = data["prediction"][model, :, j]
                error = data["prediction_numerical_sigma"][model, j]
                add_scales(columns, short, values)
                add_scales(columns, short + "fo", values / fixed_order)
                columns[short + "foerr"] = error / fixed_order[0]
                if cut == 20:
                    add_scales(columns, short + "data", values / measurements)
                    columns[short + "dataerr"] = error / measurements
            table = [v if len(v) == len(edges) else np.r_[v, v[-1]] for v in columns.values()]
            np.savetxt(args.output / f"w_asymmetry_{cut}.dat", np.column_stack(table),
                       fmt="%.12g", header=" ".join(columns), comments="")


if __name__ == "__main__":
    main()
