"""Fit a bundled real Higgs, diphoton or Z+jet sample with propagated inputs.

    python examples/application_demo.py ggh --output ggh_demo --plot
    python examples/application_demo.py aa --output aa_demo --plot
    python examples/application_demo.py zj --output zj_demo --plot

These are fresh fits to the public fixed-order tables. The extra observable is
an unconstrained prediction; the plot has no independent reference and is not
a validation of its perturbative accuracy.
"""
import argparse
from pathlib import Path
import time

import numpy as np

from maxent_upgrade import bin_diagnostics, upgrade_with_uncertainties, weight_diagnostics
from maxent_upgrade.provenance import canonical_json, file_hash


DATA = Path(__file__).resolve().parent / "data/applications"
PROCESS_NAMES = {"ggh": "Gluon-fusion Higgs", "aa": "Diphotons", "zj": "Z+jet"}


def observables(process, events):
    def spec(name, values, edges, label, title, selection=None, selection_text="original fiducial selection"):
        return dict(name=name, values=values, edges=edges, label=label, title=title,
                    selection=np.ones(len(values), dtype=bool) if selection is None else selection,
                    selection_text=selection_text)
    if process == "ggh":
        edges = [0, 2, 4, 6, 10, 20, 30, 40, 54, 70, 100, 150, 250, 400, 700]
        return [spec("recoil", events["pT_H"], edges, r"Higgs $p_T$ [GeV]", "Recoil; window rises over 54–108 GeV"),
                spec("jet", events["pT_j1_20"], [20, 30, 40, 54, 70, 100, 150, 250, 400, 700],
                     r"Leading-jet $p_T$ [GeV]", "Unconstrained jet prediction",
                     events["pT_j1_20"] > 20, "leading-jet pT>20 GeV; absolute original normalization")]
    if process == "aa":
        return [spec("recoil", events["pt_aa"], [0, 2, 4, 6, 10, 20, 28, 40, 56, 80, 120, 180, 250, 400, 700],
                     r"Diphoton $p_T$ [GeV]", "Recoil; window rises over 28–56 GeV"),
                spec("separation", events["dphi_aa"], [0, .4, .8, 1.2, 1.6, 2., 2.4, 2.6, 2.8, 3., np.pi],
                     r"Photon separation $\Delta\phi$", "Unconstrained angular prediction")]
    return [spec("second_jet", events["ptj2"], [20, 30, 40, 60, 80, 120, 200, 350, 700],
                 r"Second-jet $p_T$ [GeV]", "Jet momentum; window rises over 30–60 GeV",
                 events["ptj2"] > 20, "second-jet pT>20 GeV; absolute original normalization"),
            spec("balance", events["ptj2"] / events["ptj1"], np.linspace(0, 1.05, 15),
                 r"Jet balance $p_{T,j_2}/p_{T,j_1}$", r"Two resolved jets: $p_{T,j_2}>60$ GeV",
                 events["ptj2"] > 60, "second-jet pT>60 GeV; absolute original normalization")]


def selected_histogram(result, values, edges, selection):
    """Histogram selected events without altering or renormalizing their weights."""
    values, edges, selection = np.asarray(values), np.asarray(edges), np.asarray(selection)
    if selection.dtype != bool or selection.shape != values.shape:
        raise ValueError("selection must be a boolean mask with the observable's event order")
    # The below-range marker removes failed events from the histogram while
    # retaining the complete weight ensemble and its original normalization.
    histogram_values = np.where(selection, values, edges[0] - max(1., np.ptp(edges)))
    prediction = result.histogram(histogram_values, edges)
    prediction["selected_cross_section_pb"] = result.cross_section * result.central.weights[selection].sum()
    prediction["selected_events"] = int(selection.sum())
    return prediction, histogram_values


def plot(output, process, results, max_pull):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    from maxent_upgrade.study_report import STYLE

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.7), sharex="col",
                                 gridspec_kw=dict(height_ratios=(2.2, 1)), constrained_layout=True)
        fig.suptitle(f"{PROCESS_NAMES[process]} · propagated inputs, no independent reference", fontsize=10)
        for column, (_, prediction, label, title) in enumerate(results):
            top, ratio = axes[:, column]
            edges = prediction["edges"]
            centers = (edges[1:] + edges[:-1]) / 2
            prior, central = prediction["prior"], prediction["central"]
            top.stairs(prior, edges, baseline=None, color="#87939d", label="Prior shape at target rate")
            top.stairs(central, edges, baseline=None, color="#bd3c3c", label="Moment fit")
            if "scale_low" in prediction:
                top.stairs(prediction["scale_high"], edges, baseline=prediction["scale_low"],
                           fill=True, color="#bd3c3c", alpha=.16, label="Scale inputs")
            if "stat_low" in prediction:
                error = (prediction["stat_high"] - prediction["stat_low"]) / 2
                top.errorbar(centers, central, yerr=error, fmt="none", color="#bd3c3c",
                             capsize=2, lw=1, label="Target integration")
            valid = prior > 0
            divide = lambda x: np.divide(x, prior, out=np.full_like(prior, np.nan), where=valid)
            ratio.stairs(divide(central), edges, baseline=None, color="#bd3c3c")
            if "window_low" in prediction:
                ratio.stairs(divide(prediction["window_high"]), edges,
                             baseline=divide(prediction["window_low"]), fill=True,
                             color="#cf7a2b", alpha=.22, label="Transition inputs")
            ratio.errorbar(centers[valid], divide(central)[valid],
                           yerr=divide(prediction["sample_error"])[valid], fmt="none",
                           color="#bd3c3c", capsize=2, lw=1, label="Fixed-weight event error")
            if "pdf_low" in prediction:
                ratio.stairs(divide(prediction["pdf_low"]), edges, baseline=None,
                             color="#8871a7", ls=":", lw=1)
                ratio.stairs(divide(prediction["pdf_high"]), edges, baseline=None,
                             color="#8871a7", ls=":", lw=1, label="PDF inputs")
            ratio.axhline(1, color=".5", lw=.7)
            ratio.set(xlabel=label, ylabel="Fit / prior")
            momentum = column == 0 or process == "ggh"
            top.set(ylabel="pb / GeV" if momentum else "pb per unit", title=title)
            top.title.set_fontsize(10)
            top.set_yscale("log")
            top.legend(loc="lower left", bbox_to_anchor=(0, 1.06), frameon=False, fontsize=7, ncol=2)
            ratio.legend(loc="best", frameon=False, fontsize=6.5)
            for ax in (top, ratio):
                if momentum:
                    ax.set_xscale("log")
                    ax.set_xlim(20 if process == "zj" or (process == "ggh" and column == 1) else 2, edges[-1])
                    ax.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100, 200, 500]))
                    ax.xaxis.set_major_formatter(ScalarFormatter())
                    ax.xaxis.set_minor_formatter(NullFormatter())
                else:
                    ax.set_xlim(edges[0], edges[-1])
                ax.grid(axis="y", alpha=.15)
        fig.text(.5, -.015, f"Largest central input residual: {max_pull:.3g} integration MC errors. "
                 "Source bands are separate; local statistics are in report.json.",
                 ha="center", fontsize=8, color=".4")
        fig.savefig(output / "comparison.pdf", bbox_inches="tight")
        fig.savefig(output / "comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def run(process, output, *, prior=None, bootstrap=20, make_plot=False):
    output, prior = Path(output), Path(prior) if prior else DATA / f"{process}_prior.npz"
    output.mkdir(parents=True, exist_ok=False)
    with np.load(prior, allow_pickle=False) as source:
        events = dict(source)
    started = time.perf_counter()
    fit_checks = {}

    def check(label, fit):
        fit_checks[label] = dict(maximum_moment_pull=float(fit.max_pull) if np.isfinite(fit.max_pull) else None,
                                 maximum_relative_residual=float(fit.closure) if np.isfinite(fit.closure) else None,
                                 effective_fraction_retained=float(fit.effN_retained))
    result = upgrade_with_uncertainties(events, process, n_boot=bootstrap, seed=731132, progress=check)
    elapsed = time.perf_counter() - started
    result.provenance["input_file"] = dict(name=prior.name, sha256=file_hash(prior))
    result.notes.append("These histograms have no independent reference; input propagation is not physics validation.")
    result.notes.append("Event histogram sample errors condition on the fitted weights; use the constraint-study interface for full refit response.")
    np.savez_compressed(output / "posterior.npz", **events, **result.arrays())
    results, local, selections = [], {}, {}
    for spec in observables(process, events):
        name, values, edges = (spec[k] for k in ("name", "values", "edges"))
        selection = spec["selection"]
        prediction, histogram_values = selected_histogram(result, values, edges, selection)
        prior_weight = events["weight"] / events["weight"].sum() * result.cross_section
        prediction["prior"] = np.histogram(histogram_values, edges, weights=prior_weight)[0] / np.diff(edges)
        prediction["selection"] = np.array(spec["selection_text"])
        np.savez_compressed(output / f"{name}.npz", **prediction)
        local[name] = bin_diagnostics(values[selection], edges, result.central.weights[selection])
        selections[name] = dict(definition=spec["selection_text"], selected_events=int(selection.sum()),
                                posterior_selected_fraction=float(result.central.weights[selection].sum()),
                                posterior_unselected_fraction=float(result.central.weights[~selection].sum()))
        results.append((name, prediction, spec["label"], spec["title"]))
    if process == "ggh":
        accepted, cut = events["fid"].astype(bool), "stored photon fiducial selection after inclusive Higgs fit"
    elif process == "aa":
        accepted, cut = events["pt_sub"] > 40, "subleading-photon pT>40 GeV after the original diphoton selection"
    else:
        accepted, cut = events["ptj2"] > 60, "second-jet pT>60 GeV after the original Z+jet selection"
    acceptance = result.histogram(accepted.astype(float), [-.5, .5, 1.5], unit_area=True, density=False)
    np.savez_compressed(output / "acceptance.npz", **acceptance)
    report = dict(process=process, events=len(events["weight"]), elapsed_fit_seconds=elapsed,
                  cross_section_pb=result.cross_section, fitted_orders=result.central.chosen_moments,
                  statistics=weight_diagnostics(result.central.weights, events["weight"]), local_statistics=local,
                  observable_selections=selections,
                  input_agreement=fit_checks,
                  cut=cut, acceptance=float(acceptance["central"][1]),
                  uncertainties={source: dict(method=result.methods[source], members=len(values))
                                 for source, values in result.weights.items()},
                  notes=result.notes, provenance=result.provenance,
                  normalization="NNLO table rate for Higgs, not the separate N3LO rate used in paper Higgs plots"
                  if process == "ggh" else "released fixed-order table rate")
    (output / "report.json").write_text(canonical_json(report) + "\n")
    if make_plot:
        plot(output, process, results, result.central.max_pull)
    print(result.central.summary())
    print(f"{process}: {elapsed:.2f}s for the fit/variations; saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("process", choices=("ggh", "aa", "zj"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prior")
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    run(args.process, args.output, prior=args.prior, bootstrap=args.bootstrap, make_plot=args.plot)
