"""Fit the supplied real Drell--Yan events and compare with fixed order.

Run from a checkout after installation:
    python examples/real_drell_yan.py --output dy_demo --plot

The angular reference is an independent NNLO integration and is never fitted.
Physical disagreement in that observable is reported, not treated as a failed
software test. The recoil comparison tests the constrained hard spectrum.
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
from dy_recoil_data import measured_shape


DATA = Path(__file__).resolve().parent / "data"


def reference_summary(runs):
    mean = runs.mean(axis=0)
    return dict(central=mean[:, 0], sample_error=runs[:, :, 0].std(axis=0, ddof=1) / np.sqrt(len(runs)),
                scale_low=mean.min(axis=1), scale_high=mean.max(axis=1))


def plot_result(path, references, predictions):
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(10, 7.5), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=(2.6, 1, 1))
    recoil, angle = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])
    axes = [(recoil, figure.add_subplot(grid[1, 0], sharex=recoil)),
            (angle, figure.add_subplot(grid[1, 1], sharex=angle))]
    data_panel = figure.add_subplot(grid[2, 0])
    missing = figure.add_subplot(grid[2, 1])
    missing.axis("off")
    missing.text(.5, .5, r"No fiducial $|\cos\theta^*|$ measurement",
                 ha="center", va="center", transform=missing.transAxes)
    for column, (name, label, _) in enumerate([
        ("recoil", r"$p_T^{\ell\ell}$ [GeV]", "pT_ll"),
        ("angle", r"$|\cos\theta_{\mathrm{CS}}|$", "abs_costh"),
    ]):
        prediction, ref = predictions[name], references[name]
        edges = prediction["edges"]
        centers = (edges[:-1] + edges[1:]) / 2
        upper, lower = axes[column]
        upper.stairs(prediction["prior"], edges, color="0.5", linestyle="--", baseline=None, label="Prior, common total rate")
        upper.stairs(prediction["central"], edges, color="#b2182b", baseline=None, label="MaxEnt")
        upper.stairs(ref["central"], edges, color="black", linestyle=":", baseline=None, label="Fixed order")
        upper.errorbar(centers, ref["central"], yerr=ref["sample_error"], fmt="none",
                       color="black", lw=.7, label="Reference integration error")
        if "scale_low" in prediction:
            upper.fill_between(edges, np.r_[prediction["scale_low"], prediction["scale_low"][-1]],
                               np.r_[prediction["scale_high"], prediction["scale_high"][-1]],
                               step="post", color="#b2182b", alpha=.15, label="Propagated scale band")
        reference = ref["central"]
        positive = reference > 0
        denominator = np.where(positive, reference, np.nan)
        lower.errorbar(centers, prediction["central"] / denominator, yerr=prediction["sample_error"] / denominator,
                       fmt="o", color="#b2182b", markersize=3)
        lower.stairs(prediction["prior"] / denominator, edges, color="0.5", linestyle="--", baseline=None)
        ref_low = ref["scale_low"] / denominator
        ref_high = ref["scale_high"] / denominator
        lower.fill_between(edges, np.r_[ref_low, ref_low[-1]], np.r_[ref_high, ref_high[-1]],
                           color="0.8", step="post", label="Reference scale band")
        lower.axhline(1, color="black", lw=.6)
        lower.set_xlabel(label)
        lower.set_ylabel("/ fixed order")
        upper.set_ylabel("pb/GeV" if column == 0 else "pb per unit angle")
        upper.set_title("Constrained hard recoil" if column == 0 else "Unconstrained decay angle", pad=64)
        if column == 0:
            upper.set_yscale("log")
            lower.set_xscale("log")
        upper.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
        if name == "recoil":
            lower.set_ylim(.5, 1.6)
        else:
            lower.set_xlim(0, 1)
    measured = predictions["recoil_data"]
    edges = measured["edges"]
    select = measured["in_plot"]
    denominator = np.where(select, measured["data"], np.nan)
    widths = np.diff(edges)
    relative = measured["data_error"] / denominator
    data_panel.stairs(1 + relative, edges, baseline=1 - relative, fill=True, color=".85")
    data_panel.axhline(1, color=".5", lw=.6)
    data_panel.stairs(measured["prior"] / denominator, edges, baseline=None, ls="--", color="0.5")
    if "scale_low" in measured:
        data_panel.stairs(measured["scale_high"] / denominator, edges, baseline=measured["scale_low"] / denominator,
                          fill=True, color="#b2182b", alpha=.15)
    data_panel.errorbar((edges[:-1] + widths / 2)[select], (measured["central"] / denominator)[select],
                        yerr=(measured["sample_error"] / denominator)[select], color="#b2182b", fmt="o", ms=2.5)
    data_panel.set(xlabel=r"$p_T^{\ell\ell}$ [GeV]", ylabel="Ratio to ATLAS", xlim=(0, 500))
    figure.savefig(path)
    plt.close(figure)


def run(output, bootstrap=20, make_plot=False, prior=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    prior_path = DATA / "dy_prior.npz" if prior is None else Path(prior)
    reference_path = DATA / "dy_reference.npz"
    with np.load(prior_path, allow_pickle=False) as source:
        events = dict(source)
    required = {"weight", "mll", "y_abs", "pT_ll", "abs_costh", "pT_sub"}
    if required.difference(events):
        raise ValueError(f"comparison sample lacks {sorted(required.difference(events))}; supply selected DY arrays, including angle and subleading-lepton pT")
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
        edges = reference_data[name + "_edges"]
        references[name] = reference_summary(reference_data[name + "_per_run"])
        prediction = result.histogram(events[observable], edges)
        prior_weights = events["weight"] / events["weight"].sum() * result.cross_section
        prediction["prior"] = np.histogram(events[observable], edges, weights=prior_weights)[0] / np.diff(edges)
        prediction.update({"reference_" + k: v for k, v in references[name].items()})
        predictions[name] = prediction
        np.savez_compressed(output / (name + ".npz"), **prediction)
        diagnostics[name] = bin_diagnostics(events[observable], edges, result.central.weights)
    edges, _, data, data_error = measured_shape()
    recoil = result.histogram(events["pT_ll"], edges, unit_area=True)
    prior_density = np.histogram(events["pT_ll"], edges, weights=events["weight"])[0]
    prior_density = prior_density / (events["weight"].sum() * np.diff(edges))
    recoil.update(data=data, data_error=data_error, prior=prior_density,
                  in_plot=(edges[1:] <= 500) & np.isfinite(data) & (data > 0))
    np.savez_compressed(output / "recoil_data.npz", **recoil)
    # A tighter analysis cut reuses every coherent weight variation. This does
    # not by itself establish NNLO accuracy for the resulting acceptance.
    accepted = events["pT_sub"] > 35
    acceptance = result.histogram(accepted.astype(float), [-.5, .5, 1.5], unit_area=True, density=False)
    np.savez_compressed(output / "acceptance.npz", **acceptance)
    comparisons = {name: dict(posterior_over_reference=p["central"] / references[name]["central"],
                              reference_relative_integration_error=references[name]["sample_error"] / np.abs(references[name]["central"]))
                   for name, p in predictions.items()}
    predictions["recoil_data"] = recoil
    report = dict(comparisons=comparisons, elapsed_seconds=elapsed, events=len(events["weight"]),
                  target_cross_section_pb=result.cross_section, posterior_cross_section_pb=float(arrays["weight_upgraded"].sum()),
                  weights=weight_diagnostics(result.central.weights, events["weight"]),
                  fitted_orders=result.central.chosen_moments, local_statistics=diagnostics,
                  uncertainty_notes=result.notes, acceptance_both_leptons_above_35=float(acceptance["central"][1]),
                  reference_sha256=file_hash(reference_path),
                  reference_provenance=json.loads(str(reference_data["reference_provenance_json"])),
                  interpretation="Recoil tests the fitted hard spectrum. The independent angle is not fitted; disagreement is a physics result. No band certifies missing support or correlations.")
    (output / "report.json").write_text(json.dumps(json_value(report), indent=2) + "\n")
    if make_plot:
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
