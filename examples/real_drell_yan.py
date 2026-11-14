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


DATA = Path(__file__).resolve().parent / "data"


def reference_summary(runs):
    mean = runs.mean(axis=0)
    return dict(central=mean[:, 0], sample_error=runs[:, :, 0].std(axis=0, ddof=1) / np.sqrt(len(runs)),
                scale_low=mean.min(axis=1), scale_high=mean.max(axis=1))


def plot_result(path, references, predictions):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex="col",
                                gridspec_kw={"height_ratios": [3, 1]})
    for column, (name, label, _) in enumerate([
        ("recoil", r"$p_T^{\ell\ell}$ [GeV]", "pT_ll"),
        ("angle", r"$|\cos\theta_{\mathrm{CS}}|$", "abs_costh"),
    ]):
        prediction, ref = predictions[name], references[name]
        edges = prediction["edges"]
        centers = (edges[:-1] + edges[1:]) / 2
        upper, lower = axes[:, column]
        upper.stairs(prediction["prior"], edges, color="0.5", linestyle="--", baseline=None, label="Prior, common total rate")
        upper.stairs(prediction["central"], edges, color="#b2182b", baseline=None, label="MaxEnt")
        upper.stairs(ref["central"], edges, color="black", linestyle=":", baseline=None, label="Fixed order")
        upper.errorbar(centers, ref["central"], yerr=ref["sample_error"], fmt="none",
                       color="black", lw=.7, label="Reference integration error")
        if "scale_low" in prediction:
            upper.fill_between(edges, np.r_[prediction["scale_low"], prediction["scale_low"][-1]],
                               np.r_[prediction["scale_high"], prediction["scale_high"][-1]],
                               step="post", color="#b2182b", alpha=.15, label="Propagated scale band")
        reliable = np.ones(len(centers), dtype=bool)
        if name == "angle":
            reliable &= (edges[1:] <= .75) | (edges[:-1] >= .9)
            for ax in (upper, lower):
                ax.axvspan(.75, .9, color="0.85", alpha=.55, zorder=0,
                           label="Fixed-order shoulder" if ax is upper else None)
        lower.errorbar(centers[reliable], (prediction["central"] / ref["central"])[reliable],
                       yerr=(prediction["sample_error"] / ref["central"])[reliable],
                       fmt="o", color="#b2182b", markersize=3)
        lower.stairs(np.where(reliable, prediction["prior"] / ref["central"], np.nan),
                     edges, color="0.5", linestyle="--", baseline=None)
        ref_low = np.where(reliable, ref["scale_low"] / ref["central"], np.nan)
        ref_high = np.where(reliable, ref["scale_high"] / ref["central"], np.nan)
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
        lower.set_ylim(.5, 1.6)
    figure.tight_layout()
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
    # A tighter analysis cut reuses every coherent weight variation. This does
    # not by itself establish NNLO accuracy for the resulting acceptance.
    accepted = events["pT_sub"] > 35
    acceptance = result.histogram(accepted.astype(float), [-.5, .5, 1.5], unit_area=True, density=False)
    np.savez_compressed(output / "acceptance.npz", **acceptance)
    comparisons = {name: dict(posterior_over_reference=p["central"] / references[name]["central"],
                              reference_relative_integration_error=references[name]["sample_error"] / np.abs(references[name]["central"]))
                   for name, p in predictions.items()}
    report = dict(comparisons=comparisons, elapsed_seconds=elapsed, events=len(events["weight"]),
                  target_cross_section_pb=result.cross_section, posterior_cross_section_pb=float(arrays["weight_upgraded"].sum()),
                  weights=weight_diagnostics(result.central.weights, events["weight"]),
                  fitted_orders=result.central.chosen_moments, local_statistics=diagnostics,
                  uncertainty_notes=result.notes, acceptance_both_leptons_above_35=float(acceptance["central"][1]),
                  reference_sha256=file_hash(reference_path),
                  reference_provenance=json.loads(str(reference_data["reference_provenance_json"])),
                  angle_fixed_order_shoulder=[.75, .9],
                  interpretation="Recoil tests the fitted hard spectrum. The independent angle is not fitted; disagreement is a physics result. No band certifies missing support or correlations.")
    (output / "report.json").write_text(json.dumps(json_value(report), indent=2) + "\n")
    if make_plot:
        plot_result(output / "comparison.pdf", references, predictions)
    print(f"Saved event weights, two comparisons, a new-cut acceptance and provenance in {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="dy_demo")
    parser.add_argument("--prior", type=Path, help="another compatible selected DY sample; defaults to the committed Pythia subset")
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    run(args.output, args.bootstrap, args.plot, prior=args.prior)
