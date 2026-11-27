"""Fit the supplied real Drell--Yan events and compare with fixed order.

Run from a checkout after installation:
    python examples/real_drell_yan.py --output dy_demo --plot

Compare the hard recoil and lepton acoplanarity with fixed order and ATLAS.
The fit uses moments of the dilepton mass, rapidity and transverse momentum.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maxent_upgrade import bin_diagnostics, upgrade_with_uncertainties, weight_diagnostics
from maxent_upgrade.provenance import file_hash, json_value
from dy_recoil_data import DATA as MEASUREMENTS, measured_shape, phistar_reference


DATA = Path(__file__).resolve().parent / "data"


def reference_summary(runs):
    mean = runs.mean(axis=0)
    return dict(central=mean[:, 0], sample_error=runs[:, :, 0].std(axis=0, ddof=1) / np.sqrt(len(runs)),
                scale_low=mean.min(axis=1), scale_high=mean.max(axis=1))


def shape(result, events, observable, edges):
    """Use the full fiducial rate of each variation, including events outside the bins."""
    values = events[observable]
    out = result.histogram(values, edges, unit_area=True)
    widths = np.diff(edges)
    base = events["weight"]
    out["prior"] = np.histogram(values, edges, weights=base)[0] / (base.sum() * widths)
    out["scale_members"] = np.vstack([out["central"], *[
        np.histogram(values, edges, weights=weights)[0] / widths for weights in result.weights["scale"]]])
    return out


def plot_result(path, references, predictions):
    from plot_dy_portability import (COLORS, draw, finish, fixed_order, make_figure, measurement,
                                     prediction_ratio, reference_spectra)
    import matplotlib.pyplot as plt
    figure, axes = make_figure()
    color = COLORS["pythia"]
    reference_spectra(axes, MEASUREMENTS, references["recoil_shape"])
    for col, name in enumerate(("recoil", "phistar")):
        fit = predictions[name + "_shape"]
        ref = references[name + "_shape"]
        measured = predictions[name + "_data"]
        edges = measured["edges"]
        denom = measurement(axes[0, col], axes[2, col], edges, measured["data"], measured["data_error"],
                            500 if name == "recoil" else 10)
        draw(axes[0, col], edges, measured["prior"], color, dashed=True)
        draw(axes[0, col], edges, measured["central"], color)
        fixed_order(axes[1, col], ref)
        draw(axes[1, col], fit["edges"], fit["prior"] / ref["central"], color, dashed=True)
        error = fit["sample_error"]
        if "stat_high" in fit:
            error = np.hypot(error, fit["stat_high"] - fit["central"])
        prediction_ratio(axes[1, col], fit["edges"], fit["scale_members"], ref["scale_members"], color, error)
        draw(axes[2, col], edges, measured["prior"] / denom, color, dashed=True)
        draw(axes[2, col], edges, measured["central"] / denom, color, error=measured["sample_error"] / denom,
             band=(measured["scale_low"] / denom, measured["scale_high"] / denom))
    finish(figure, axes, [("Shower", color)])
    figure.savefig(path)
    plt.close(figure)


def run(output, bootstrap=20, make_plot=False, prior=None):
    output = Path(output)
    prior_path = DATA / "dy_prior.npz" if prior is None else Path(prior)
    reference_path = DATA / "dy_reference.npz"
    with np.load(prior_path, allow_pickle=False) as source:
        events = dict(source)
    required = {"weight", "mll", "y_abs", "pT_ll", "pT_sub"}
    if make_plot:
        required.add("phistar")
    missing = required.difference(events)
    if missing:
        raise ValueError(f"Comparison requires the event columns {sorted(missing)}")
    output.mkdir(parents=True, exist_ok=False)
    with np.load(reference_path, allow_pickle=False) as source:
        reference_data = dict(source)
    start = time.perf_counter()
    result = upgrade_with_uncertainties(
        events, "dy", n_boot=bootstrap,
        progress=lambda label, fit: print(f"{label}: {fit.summary()}", flush=True))
    elapsed = time.perf_counter() - start
    result.provenance["input_file"] = dict(name=prior_path.name, sha256=file_hash(prior_path))
    arrays = result.arrays()
    np.savez_compressed(output / "posterior.npz", **events, **arrays)
    predictions, references, diagnostics = {}, {}, {}
    for name, observable in [("recoil", "pT_ll"), ("angle", "abs_costh")]:
        if observable not in events:
            continue
        edges = reference_data[name + "_edges"]
        references[name] = reference_summary(reference_data[name + "_per_run"])
        prediction = result.histogram(events[observable], edges)
        prior_weights = events["weight"] / events["weight"].sum() * result.cross_section
        prediction["prior"] = np.histogram(events[observable], edges, weights=prior_weights)[0] / np.diff(edges)
        prediction.update({"reference_" + k: v for k, v in references[name].items()})
        predictions[name] = prediction
        np.savez_compressed(output / (name + ".npz"), **prediction)
        diagnostics[name] = bin_diagnostics(events[observable], edges, result.central.weights)
    with np.load(DATA / "dy_portability/recoil_reference.npz") as source:
        references["recoil_shape"] = dict(source)
    comparisons = {name: dict(posterior_over_reference=fit["central"] / references[name]["central"],
                              reference_relative_integration_error=references[name]["sample_error"] / references[name]["central"])
                   for name, fit in predictions.items()}
    for name, observable in [("recoil", "pT_ll"), ("phistar", "phistar")]:
        if observable not in events:
            continue
        if name == "phistar":
            references["phistar_shape"] = phistar_reference()
        ref = references[name + "_shape"]
        fit = shape(result, events, observable, ref["edges"])
        fit.update({"reference_" + k: v for k, v in ref.items() if k != "edges"})
        predictions[name + "_shape"] = fit
        np.savez_compressed(output / (name + "_shape.npz"), **fit)
        comparisons[name + "_shape"] = dict(posterior_over_reference=fit["central"] / ref["central"])
        edges, _, data, data_error = measured_shape(observable=name)
        measured = shape(result, events, observable, edges)
        measured.update(data=data, data_error=data_error,
                        in_plot=(edges[1:] <= (500 if name == "recoil" else 10)) & np.isfinite(data) & (data > 0))
        predictions[name + "_data"] = measured
        np.savez_compressed(output / (name + "_data.npz"), **measured)
    # A tighter analysis cut reuses every coherent weight variation. This does
    # not by itself establish NNLO accuracy for the resulting acceptance.
    accepted = events["pT_sub"] > 35
    acceptance = result.histogram(accepted.astype(float), [-.5, .5, 1.5], unit_area=True, density=False)
    np.savez_compressed(output / "acceptance.npz", **acceptance)
    report = dict(comparisons=comparisons, elapsed_seconds=elapsed, events=len(events["weight"]),
                  target_cross_section_pb=result.cross_section, posterior_cross_section_pb=float(arrays["weight_upgraded"].sum()),
                  weights=weight_diagnostics(result.central.weights, events["weight"]),
                  fitted_orders=result.central.chosen_moments, local_statistics=diagnostics,
                  uncertainty_notes=result.notes, acceptance_both_leptons_above_35=float(acceptance["central"][1]),
                  reference_sha256=file_hash(reference_path),
                  reference_provenance=json.loads(str(reference_data["reference_provenance_json"])),
                  interpretation="Hard recoil tests the fitted spectrum. Lepton acoplanarity uses the directions of both leptons and is predicted with the same event weights.")
    (output / "report.json").write_text(json.dumps(json_value(report), indent=2) + "\n")
    if make_plot:
        from matplotlib import pyplot as plt
        from plot_dy_portability import STYLE
        with plt.rc_context(STYLE):
            plot_result(output / "comparison.pdf", references, predictions)
    print(f"Saved event weights, reference comparisons, a new-cut acceptance and provenance in {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="dy_demo")
    parser.add_argument("--prior", type=Path, help="another compatible selected DY sample; defaults to the committed Pythia subset")
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    run(args.output, args.bootstrap, args.plot, prior=args.prior)
