"""Redraw the full-sample generator comparison from compact numerical caches.

    python examples/plot_dy_portability.py --data examples/data/dy_portability --output dy_portability_plots

No event fitting runs here. Use real_drell_yan.py for a fresh fit on either
committed compact prior; those subsets have different event statistics.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from dy_recoil_data import measured_shape, read_bins


COLORS = {"pythia": "#b13c35", "herwig": "#285f96", "reference": "#373737"}


def load(directory, name):
    with np.load(Path(directory) / (name + ".npz"), allow_pickle=False) as source:
        return dict(source)


def ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator != 0)


def draw(ax, edges, values, color, *, dashed=False, valid=None, band=None, error=None, offset=0):
    """Draw native bins without connecting gaps or inventing interpolated values."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) if valid is None else np.asarray(valid) & np.isfinite(values)
    style = ":" if color == COLORS["reference"] else ("--" if dashed else "-")
    ax.stairs(np.where(valid, values, np.nan), edges, baseline=None, color=color, lw=1.3, linestyle=style)
    if band is not None:
        low, high = (np.where(valid, np.asarray(v), np.nan) for v in band)
        x = np.repeat(edges, 2)[1:-1]
        ax.fill_between(x, np.repeat(low, 2), np.repeat(high, 2), color=color, alpha=.16, linewidth=0)
    if error is not None:
        centers = (edges[:-1] + edges[1:]) / 2 + offset * np.diff(edges)
        ax.errorbar(centers[valid], values[valid], yerr=np.asarray(error)[valid],
                    color=color, fmt=".", ms=2.7, lw=.7, capsize=0)


def plot_distribution(top, comparison, directory, name):
    """Use the same native fixed-order bins for the spectrum and its ratios."""
    ref = load(directory, name + "_reference")
    edges = ref["edges"]
    reference = ref["central"]
    valid = np.isfinite(reference) & (reference > 0)
    draw(top, edges, reference, COLORS["reference"], valid=valid, band=(ref["scale_low"], ref["scale_high"]))
    ref_band = (ratio(ref["scale_low"], reference), ratio(ref["scale_high"], reference))
    ref_error = ratio(ref["sample_error"], reference)
    extent = [ref_band[0][valid], ref_band[1][valid], 1 - ref_error[valid], 1 + ref_error[valid]]
    draw(comparison, edges, np.ones_like(reference), COLORS["reference"], valid=valid,
         error=ref_error, band=ref_band)
    comparison.axhline(1, color=".6", lw=.6, zorder=0)
    peak = None
    if name == "recoil":
        peak = top.inset_axes([.49, .57, .46, .37])
        peak.set(xlim=(0, 30), xlabel=r"$q_T$ [GeV]")
        peak.tick_params(direction="in", top=True, right=True, labelsize=8)
        peak.xaxis.label.set_size(8)
    for slot, label in enumerate(("pythia", "herwig")):
        h = load(directory, f"{label}_{name}")
        np.testing.assert_array_equal(edges, h["edges"])
        color = COLORS[label]
        if name == "recoil":
            # The low and hard histograms have separate native edges. Drawing
            # them in two pieces preserves the gap from 60 to 60.615 GeV.
            full = load(directory, label + "_full_recoil")
            keep = np.count_nonzero(full["edges"][1:] <= edges[0])
            for key in ("prior", "central"):
                draw(top, full["edges"][:keep + 1], full[key][:keep], color, dashed=key == "prior")
                soft = np.count_nonzero(full["edges"][1:] <= 30)
                draw(peak, full["edges"][:soft + 1], full[key][:soft], color, dashed=key == "prior")
        draw(top, edges, h["prior"], color, dashed=True)
        draw(top, edges, h["central"], color)
        members = np.vstack([h["central"], h["scale_members"]])
        if members.shape != ref["scale_members"].shape:
            raise ValueError("posterior and reference need the same seven ordered scale members")
        scales = ratio(members, ref["scale_members"])
        error = ratio(np.hypot(h["sample_error"], h["stat_members"].std(axis=0, ddof=1)), reference)
        prior_ratio = ratio(h["prior"], reference)
        central_ratio = ratio(h["central"], reference)
        extent.extend([prior_ratio[valid], (central_ratio - error)[valid], (central_ratio + error)[valid],
                       scales[:, valid].ravel()])
        draw(comparison, edges, prior_ratio, color, dashed=True, valid=valid)
        draw(comparison, edges, central_ratio, color, valid=valid,
             band=(scales.min(axis=0), scales.max(axis=0)), error=error,
             offset=(-.08 if slot == 0 else .08))
    finite = np.concatenate(extent)
    low, high = np.nanmin(finite), np.nanmax(finite)
    margin = max(.03, .10 * (high - low))
    comparison.set_ylim(low - margin, high + margin)
    comparison.set_ylabel("Ratio to fixed order")
    return peak


def plot_measured_recoil(spectrum, peak, comparison, directory):
    edges, rows, data, error = measured_shape(directory)
    keep = (edges[1:] <= 500) & (data > 0)
    centers = (edges[:-1] + edges[1:]) / 2
    for ax, selected in [(spectrum, keep), (peak, keep & (edges[1:] <= 30))]:
        ax.errorbar(centers[selected], data[selected], xerr=np.diff(edges)[selected] / 2,
                    yerr=error[selected], fmt="o", color=".15", ms=2.8, lw=.6)
    relative = ratio(error, data)
    comparison.stairs(np.where(keep, 1 + relative, np.nan), edges,
                      baseline=np.where(keep, 1 - relative, np.nan), fill=True, color=".90")
    comparison.axhline(1, color=".5", lw=.6)
    draw(comparison, edges, rows["rew_over_data"], COLORS["pythia"], valid=keep,
         error=ratio(rows["rew_event_error_GeV_inv"], data),
         band=(ratio(rows["rew_scale_low_GeV_inv"], data), ratio(rows["rew_scale_high_GeV_inv"], data)))
    common_edges, common = read_bins(Path(directory) / "atlas_recoil_common_integrals.csv")
    members = np.array([common[f"herwig_scale_{i}_probability"] for i in range(7)])
    ratios = members / common["data_probability"]
    draw(comparison, common_edges, ratios[0], COLORS["herwig"], band=(ratios.min(axis=0), ratios.max(axis=0)))
    comparison.set(xlim=(0, 500), xlabel=r"$q_T$ [GeV]", ylabel="Ratio to ATLAS")
    limits = [rows["rew_over_data"] - ratio(rows["rew_event_error_GeV_inv"], data),
              rows["rew_over_data"] + ratio(rows["rew_event_error_GeV_inv"], data),
              ratio(rows["rew_scale_low_GeV_inv"], data), ratio(rows["rew_scale_high_GeV_inv"], data),
              1 - relative, 1 + relative]
    shown = np.r_[np.asarray(limits)[:, keep].ravel(), ratios.ravel()]
    margin = (shown.max() - shown.min()) * .1
    comparison.set_ylim(shown.min() - margin, shown.max() + margin)


def plot(directory):
    fig = plt.figure(figsize=(9.6, 7.0))
    grid = fig.add_gridspec(3, 2, height_ratios=(1.8, 1, 1), hspace=.48, wspace=.30)
    recoil = fig.add_subplot(grid[0, 0])
    angle = fig.add_subplot(grid[0, 1])
    hard = fig.add_subplot(grid[1, 0])
    measured = fig.add_subplot(grid[2, 0])
    angular = fig.add_subplot(grid[1, 1], sharex=angle)
    missing = fig.add_subplot(grid[2, 1])
    missing.axis("off")
    missing.text(.5, .5, r"No fiducial $|\cos\theta^*|$ measurement",
                 ha="center", va="center", transform=missing.transAxes)
    peak = plot_distribution(recoil, hard, directory, "recoil")
    plot_distribution(angle, angular, directory, "angle")
    plot_measured_recoil(recoil, peak, measured, Path(directory).parent / "paper_distributions/drell_yan")

    recoil.set(xlim=(0, 500), yscale="log", ylim=(1e-6, .2),
               ylabel=r"$\sigma^{-1}d\sigma/dq_T$ [GeV$^{-1}$]", xlabel=r"$q_T$ [GeV]")
    angle.set(xlim=(0, 1), ylabel=r"$\sigma^{-1}d\sigma/d|\cos\theta^{*}|$",
              xlabel=r"$|\cos\theta^{*}|$")
    angle.set_ylim(bottom=0)
    hard.set(xlim=(60, 500), xlabel=r"$q_T$ [GeV]")
    hard.set_xticks([60, 150, 250, 350, 500])
    angular.set_xlabel(r"$|\cos\theta^{*}|$")
    for ax in fig.axes:
        ax.tick_params(direction="in", top=True, right=True)
    handles = [Line2D([], [], color=COLORS[label], linestyle=style, label=text)
               for label, style, text in [("pythia", "--", "MG5+Pythia prior"), ("herwig", "--", "Herwig prior"),
                                          ("reference", ":", "Fixed order"), ("pythia", "-", "MaxEnt, MG5+Pythia"),
                                          ("herwig", "-", "MaxEnt, Herwig")]]
    handles.extend([Line2D([], [], marker="o", linestyle="none", color=".15", label="ATLAS 13 TeV"),
                    Patch(facecolor=".9", label="ATLAS error in measured bins")])
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=.87, bottom=.10, left=.09, right=.98)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(Path(__file__).resolve().parent / "data/dy_portability"))
    parser.add_argument("--output", default="dy_portability_plots")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans", "font.size": 10}):
        fig = plot(args.data)
        for suffix in ("pdf", "png"):
            fig.savefig(output / ("dy_portability." + suffix), dpi=180)
        plt.close(fig)
    print(f"Saved {output}.")


if __name__ == "__main__":
    main()
