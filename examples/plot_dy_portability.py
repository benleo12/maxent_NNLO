"""Plot the full Drell–Yan samples against fixed order and ATLAS.

    python examples/plot_dy_portability.py --output dy_portability_plots

Use real_drell_yan.py to fit and plot either of the supplied compact event samples.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from dy_recoil_data import measured_shape, phistar_reference, read_bins


COLORS = {"pythia": "#ad392f", "herwig": "#28649a", "reference": "#373737"}
STYLE = {"font.family": "serif", "mathtext.fontset": "stix", "axes.linewidth": .75, "pdf.fonttype": 42}
LABELS = (r"$q_T$ [GeV]", r"$\phi^*_\eta$")


def load(directory, name):
    with np.load(Path(directory) / (name + ".npz"), allow_pickle=False) as source:
        return dict(source)


def draw(ax, edges, values, color, *, dashed=False, band=None, error=None, offset=0):
    style = ":" if color == COLORS["reference"] else ("--" if dashed else "-")
    ax.stairs(values, edges, baseline=None, color=color, lw=1.35, linestyle=style)
    if band is not None:
        ax.fill_between(np.repeat(edges, 2)[1:-1], np.repeat(band[0], 2), np.repeat(band[1], 2),
                        color=color, alpha=.16, linewidth=0)
    if error is not None:
        centers = edges[:-1] + (.5 + offset) * np.diff(edges)
        ax.errorbar(centers, values, yerr=error, color=color, fmt=".", ms=2.6, lw=.7)


def fixed_order(ax, ref):
    central = ref["central"]
    draw(ax, ref["edges"], np.ones_like(central), COLORS["reference"],
         band=(ref["scale_low"] / central, ref["scale_high"] / central),
         error=ref["sample_error"] / central)


def prediction_ratio(ax, edges, fit_members, ref_members, color, error, offset=0):
    """Vary numerator and denominator at the same scales."""
    if fit_members.shape != ref_members.shape:
        raise ValueError("The fit and reference need the same ordered scale choices and bins")
    ratios = fit_members / ref_members
    draw(ax, edges, ratios[0], color, band=(ratios.min(0), ratios.max(0)),
         error=error / ref_members[0], offset=offset)


def measurement(top, lower, edges, density, error, x_max):
    keep = np.isfinite(density) & (density > 0) & (edges[1:] <= x_max)
    widths = np.diff(edges)
    top.errorbar((edges[:-1] + widths / 2)[keep], density[keep], xerr=widths[keep] / 2,
                 yerr=error[keep], fmt="o", color=".15", ms=2.6, lw=.6, zorder=5)
    relative = np.divide(error, density, out=np.full_like(error, np.nan), where=keep)
    lower.stairs(1 + relative, edges, baseline=1 - relative, fill=True, color=".90")
    return np.where(keep, density, np.nan)


def reference_spectra(axes, directory, recoil_ref):
    draw(axes[0, 0], recoil_ref["edges"], recoil_ref["central"], COLORS["reference"],
         band=(recoil_ref["scale_low"], recoil_ref["scale_high"]))
    edges, rows = read_bins(Path(directory) / "phistar_reference_spectrum.csv", ("low", "high"))
    members = np.array([rows[f"fixed_order_scale_{i}"] for i in range(7)])
    draw(axes[0, 1], edges, members[0], COLORS["reference"], band=(members.min(0), members.max(0)))


def make_figure():
    return plt.subplots(3, 2, figsize=(9, 7.1), gridspec_kw={"height_ratios": (1.9, 1, 1)})


def ratio_limits(axis):
    """Crop to the physical predictions, uncertainty bands and error-bar tips."""
    values = [np.asarray([1.])]
    for patch in axis.patches:
        steps = patch.get_data()
        values.append(steps.values)
        if steps.baseline is not None:
            values.append(np.atleast_1d(steps.baseline))
    for collection in axis.collections:
        values.extend(path.vertices[:, 1] for path in collection.get_paths())
    all_values = np.concatenate(values)
    low, high = np.nanmin(all_values), np.nanmax(all_values)
    margin = max(.015, .1 * (high - low))
    axis.set_ylim(low - margin, high + margin)


def finish(figure, axes, generators):
    axes[0, 0].set(xlim=(0, 500), yscale="log", ylabel=r"$\sigma^{-1}d\sigma/dq_T$ [GeV$^{-1}$]")
    axes[0, 1].set(xscale="log", yscale="log", xlim=(.001, 10),
                   ylabel=r"$\sigma^{-1}d\sigma/d\phi^*_\eta$")
    axes[1, 0].set_xlim(60, 500)
    axes[1, 0].set_xticks([60, 150, 250, 350, 500])
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlim(axes[1, 1].dataLim.xmin, axes[1, 1].dataLim.xmax)
    axes[2, 0].set_xlim(0, 500)
    axes[2, 1].set(xscale="log", xlim=(.001, 10))
    for column, xlabel in enumerate(LABELS):
        for axis in axes[:, column]:
            axis.set_xlabel(xlabel)
            axis.tick_params(direction="in", top=True, right=True)
        for axis, reference in zip(axes[1:, column], ["fixed order", "ATLAS"]):
            axis.axhline(1, color=".55", lw=.65, zorder=0)
            axis.set_ylabel("Ratio to " + reference)
            ratio_limits(axis)
    handles = []
    for label, color in generators:
        handles += [Line2D([], [], color=color, ls="--", label=label + " prior"),
                    Line2D([], [], color=color, label=label + " reweighted")]
    handles += [Line2D([], [], color=COLORS["reference"], ls=":", label="Fixed order"),
                Line2D([], [], color=".15", marker="o", ls="none", label="ATLAS 13 TeV"),
                Patch(facecolor=".90", label="Experimental uncertainty")]
    figure.legend(handles=handles, loc="upper center", ncol=4 if len(generators) == 2 else 3,
                  frameon=False, fontsize=9)
    figure.subplots_adjust(top=.90, bottom=.08, left=.095, right=.98, hspace=.54, wspace=.32)


def plot(directory):
    data_dir = Path(directory).parent / "paper_distributions/drell_yan"
    fig, axes = make_figure()
    recoil_ref = load(directory, "recoil_reference")
    phi_ref = phistar_reference(data_dir)
    phi_edges, phi_rows = read_bins(data_dir / "phistar_fixed_order.csv", ("low", "high"))
    reference_spectra(axes, data_dir, recoil_ref)
    fixed_order(axes[1, 0], recoil_ref)
    fixed_order(axes[1, 1], phi_ref)
    peak = axes[0, 0].inset_axes([.48, .55, .47, .39])
    peak.set(xlim=(0, 30), xlabel=LABELS[0])
    peak.tick_params(direction="in", top=True, right=True, labelsize=7)
    peak.xaxis.label.set_size(7)
    for slot, gen in enumerate(("pythia", "herwig")):
        color = COLORS[gen]
        offset = -.09 if slot == 0 else .09
        hard = load(directory, gen + "_recoil")
        full = load(directory, gen + "_full_recoil")
        soft = np.count_nonzero(full["edges"][1:] <= hard["edges"][0])
        np.testing.assert_array_equal(hard["edges"], recoil_ref["edges"])
        phi = load(directory, gen + "_phistar")
        for key in ("prior", "central"):
            draw(axes[0, 0], full["edges"][:soft + 1], full[key][:soft], color, dashed=key == "prior")
            n_peak = np.count_nonzero(full["edges"][1:] <= 30)
            draw(peak, full["edges"][:n_peak + 1], full[key][:n_peak], color, dashed=key == "prior")
            draw(axes[0, 0], hard["edges"], hard[key], color, dashed=key == "prior")
            if gen == "herwig" or key == "prior":
                draw(axes[0, 1], phi["edges"], phi[key], color, dashed=key == "prior")
        draw(axes[1, 0], hard["edges"], hard["prior"] / recoil_ref["central"], color, dashed=True)
        members = np.vstack([hard["central"], hard["scale_members"]])
        error = np.hypot(hard["sample_error"], hard["stat_members"].std(0, ddof=1))
        prediction_ratio(axes[1, 0], hard["edges"], members, recoil_ref["scale_members"], color, error, offset)
        phi_members = np.stack([phi_rows[f"{gen}_scale_{i}"] for i in range(7)])
        phi_error = np.hypot(phi_rows[gen + "_event_error"], phi_rows[gen + "_fit_input_error"])
        prediction_ratio(axes[1, 1], phi_edges, phi_members, phi_ref["scale_members"], color, phi_error, offset)

    for col, name in enumerate(("recoil", "phistar")):
        edges, rows, data, err = measured_shape(data_dir, name)
        den = measurement(axes[0, col], axes[2, col], edges, data, err, 500 if col == 0 else 10)
        if name == "recoil":
            keep = edges[1:] <= 30
            peak.errorbar(((edges[:-1] + edges[1:]) / 2)[keep], data[keep],
                          xerr=np.diff(edges)[keep] / 2, yerr=err[keep], fmt="o", color=".15", ms=2.1, lw=.5)
            draw(axes[2, 0], edges, rows["prior_GeV_inv"] / den, COLORS["pythia"], dashed=True)
            central, sample = rows["rew_GeV_inv"], rows["rew_event_error_GeV_inv"]
            bands = (rows["rew_scale_low_GeV_inv"], rows["rew_scale_high_GeV_inv"])
        else:
            central, sample = rows["prediction"], rows["prediction_event_error"]
            draw(axes[0, 1], edges, central, COLORS["pythia"])
            bands = (rows["scale_min"], rows["scale_max"])
        draw(axes[2, col], edges, central / den, COLORS["pythia"], error=sample / den,
             band=(bands[0] / den, bands[1] / den))
    edges, rows = read_bins(data_dir / "atlas_recoil_common_integrals.csv")
    ratios = np.stack([rows[f"herwig_scale_{i}_probability"] for i in range(7)]) / rows["data_probability"]
    draw(axes[2, 0], edges, rows["herwig_prior_probability"] / rows["data_probability"], COLORS["herwig"], dashed=True)
    draw(axes[2, 0], edges, ratios[0], COLORS["herwig"], band=(ratios.min(0), ratios.max(0)))
    edges, rows = read_bins(data_dir / "herwig_atlas_groups.csv", ("low", "high"))
    den = rows["data_probability"]
    low = np.min([rows[f"herwig_scale_{i}_low"] for i in range(7)], axis=0) / den
    high = np.max([rows[f"herwig_scale_{i}_high"] for i in range(7)], axis=0) / den
    draw(axes[2, 1], edges, rows["herwig_scale_0_probability"] / den, COLORS["herwig"],
         band=(low, high), error=np.hypot(rows["herwig_event_error"], rows["herwig_input_mc_error"]) / den)
    for ax, source, field in [(axes[1, 1], "prior_phistar_fo.csv", "prior_over_fixed_order"),
                              (axes[2, 1], "prior_phistar_atlas.csv", "prior_over_data")]:
        priors = np.genfromtxt(data_dir / source, names=True, delimiter=",", dtype=None, encoding="utf-8")
        for gen in ("pythia", "herwig"):
            selected = priors[priors["generator"] == gen]
            edges = np.r_[selected["low"], selected["high"][-1]]
            draw(ax, edges, selected[field], COLORS[gen], dashed=True)
    finish(fig, axes, [("MG5+Pythia", COLORS["pythia"]), ("Herwig", COLORS["herwig"])])
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(Path(__file__).resolve().parent / "data/dy_portability"))
    parser.add_argument("--output", default="dy_portability_plots")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with plt.rc_context(STYLE):
        fig = plot(args.data)
        for suffix in ("pdf", "png"):
            fig.savefig(output / ("dy_portability." + suffix), dpi=180)
        plt.close(fig)
    print(f"Saved {output}.")


if __name__ == "__main__":
    main()
