"""Export the conditional lepton acceptance from the independent W validation release.

Usage: export_acceptance.py PATH_TO_examples/data/w_predictivity/full_study
Quantities are in percent of the indicated denominator, or percentage points.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(data, metric):
    meta = json.loads(str(data["metadata"]))
    column = meta["labels"].index(metric)
    central = 0
    models = ["R", "R+H+S"]
    selected, augmented = (meta["models"].index(name) for name in models)
    pair = meta["pairs"].index([models[1], models[0]])
    reference = data["reference"][central, column]
    reference_variance = data["reference_covariance"][column, column]
    row = {"reference": reference, "reference_error": np.sqrt(reference_variance)}
    means = data["means"][:, central, column]
    prior = meta["models"].index("P")
    for name, index in [("without", selected), ("with", augmented), ("prior", prior)]:
        var = (data["covariance_numerical"][index, column, column]
               + data["covariance_prior"][index, column, column])
        row[name] = means[index]
        row[f"{name}_error"] = np.sqrt(var)
        row[f"{name}_minus_ref"] = means[index] - reference
        row[f"{name}_minus_ref_error"] = np.sqrt(var + reference_variance)
    row["paired_change"] = means[augmented] - means[selected]
    pair_var = (data["paired_covariance_numerical"][pair, column, column]
                + data["paired_covariance_prior"][pair, column, column])
    row["paired_change_error"] = np.sqrt(pair_var)
    return {name: 100 * float(value) for name, value in row.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path)
    p.add_argument("--output", type=Path, default=Path("."))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    audit = {}
    denominator = {"pt32": "baseline", "pt32_given_hard": "baseline_qTgt50"}
    for charge_number, charge in enumerate(("WP", "WM"), start=1):
        path = args.inputs / f"{charge}_metrics.npz"
        with np.load(path, allow_pickle=False) as data:
            metrics = {metric: summarize(data, metric) for metric in
                       ("hard", "pt28", "pt32", "pt32_given_hard", "old_pt27", "mt60", "combined")}
            for key, denom in denominator.items():
                rows.append({"charge": charge, "denominator": denom, "charge_number": charge_number,
                             "hard_subset": int(key == "pt32_given_hard"), "threshold_GeV": 32, **metrics[key]})
            audit[charge] = {"input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "metrics_in_percent": metrics}
    path = args.output / "w_acceptance_32.csv"
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.12g}" if isinstance(value, float) else value
                             for key, value in row.items()})
    (args.output / "acceptance_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
