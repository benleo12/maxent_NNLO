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
    style = "--" if dashed else "-"
    ax.stairs(np.where(valid, values, np.nan), edges, baseline=None, color=color, lw=1.3, linestyle=style)
    if band is not None:
        low, high = (np.where(valid, np.asarray(v), np.nan) for v in band)
        ax.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]],
                        step="post", color=color, alpha=.13, linewidth=0)
    if error is not None:
        centers = (edges[:-1] + edges[1:]) / 2 + offset * np.diff(edges)
        ax.errorbar(centers[valid], values[valid], yerr=np.asarray(error)[valid],
                    color=color, fmt=".", ms=2.7, lw=.7, capsize=0)


def limits(arrays, low, high):
    values = np.concatenate([np.asarray(a).ravel() for a in arrays])
    values = values[np.isfinite(values)]
    if len(values):
        return min(low, float(values.min()) * .95), max(high, float(values.max()) * 1.05)
    return low, high


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(Path(__file__).resolve().parent / "data/dy_portability"))
    parser.add_argument("--output", default="dy_portability_plots")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(3, 2, figsize=(11.2, 7.6), sharex="col",
                                 gridspec_kw={"height_ratios": [2.2, 1.2, 1], "hspace": .07, "wspace": .32})
        for column, name in enumerate(("recoil", "angle")):
            top, middle, bottom = axes[:, column]
            ref = load(args.data, name + "_reference")
            edges = ref["edges"]
            centers = (edges[:-1] + edges[1:]) / 2
            valid = np.ones(len(centers), dtype=bool) if name == "recoil" else ((centers < .75) | (centers > .9))
            shade = (30, 60) if name == "recoil" else (.75, .9)
            for ax in axes[:, column]:
                ax.axvspan(*shade, color=".92", zorder=0)
                ax.tick_params(direction="in", top=True, right=True)
            draw(top, edges, ref["central"], COLORS["reference"], valid=valid)
            draw(middle, edges, np.ones(len(centers)), COLORS["reference"], valid=valid,
                 error=ratio(ref["sample_error"], ref["central"]),
                 band=(ratio(ref["scale_low"], ref["central"]), ratio(ref["scale_high"], ref["central"])))
            middle.axhline(1, color=".6", lw=.6)
            bottom.axhline(1, color=".6", lw=.6)
            ratios, changes = [], []
            for slot, label in enumerate(("pythia", "herwig")):
                hist = load(args.data, label + "_" + name)
                full = load(args.data, label + "_full_recoil") if name == "recoil" else hist
                np.testing.assert_array_equal(edges, hist["edges"])
                if name == "recoil":
                    # Every hard curve uses the reference's exact native bins.
                    # The low cache ends at 60 GeV; the hard cache begins at
                    # 60.615 GeV. Leave their uncomputed interval empty.
                    count = int(np.count_nonzero(full["edges"][1:] <= edges[0]))
                    display = {"edges": np.r_[full["edges"][:count + 1], edges]}
                    for key in ("prior", "central", "scale_low", "scale_high"):
                        display[key] = np.r_[full[key][:count], np.nan, hist[key]]
                    full = display
                color = COLORS[label]
                draw(top, full["edges"], full["prior"], color, dashed=True)
                draw(top, full["edges"], full["central"], color)
                central_ratio = ratio(hist["central"], ref["central"])
                prior_ratio = ratio(hist["prior"], ref["central"])
                # A scale member is a complete normalized prediction. Pair its
                # numerator and reference at that same scale.
                members = np.vstack([hist["central"], hist["scale_members"]])
                if members.shape != ref["scale_members"].shape:
                    raise ValueError("posterior and reference need the same seven ordered scale members")
                scale_ratios = ratio(members, ref["scale_members"])
                integration_error = hist["stat_members"].std(axis=0, ddof=1)
                error = np.hypot(hist["sample_error"], integration_error)
                draw(middle, edges, prior_ratio, color, dashed=True, valid=valid)
                draw(middle, edges, central_ratio, color, valid=valid, offset=(-.08 if slot == 0 else .08),
                     error=ratio(error, ref["central"]), band=(scale_ratios.min(axis=0), scale_ratios.max(axis=0)))
                change = ratio(full["central"], full["prior"])
                draw(bottom, full["edges"], change, color,
                     band=(ratio(full["scale_low"], full["prior"]), ratio(full["scale_high"], full["prior"])))
                ratios.extend([central_ratio[valid], prior_ratio[valid]])
                changes.append(change)
            middle.set_ylim(*limits(ratios, .7, 1.3))
            bottom.set_ylim(*limits(changes, .8, 1.2))
            middle.set_ylabel("Ratio to fixed order")
            bottom.set_ylabel("Ratio to own prior")
        for ax in axes[:, 0]:
            ax.set_xscale("log")
            ax.set_xlim(.5, 500)
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_ylim(1e-6, .2)
        axes[0, 0].set_ylabel(r"$\sigma^{-1}d\sigma/dq_T$ [GeV$^{-1}$]")
        axes[2, 0].set_xlabel(r"$q_T$ [GeV]")
        axes[0, 1].set_ylim(bottom=0)
        axes[0, 1].set_ylabel(r"$\sigma^{-1}d\sigma/d|\cos\theta^{*}|$")
        axes[2, 1].set_xlabel(r"$|\cos\theta^{*}|$")
        axes[2, 1].set_xlim(0, 1)
        handles = [Line2D([], [], color=COLORS[label], linestyle=style, label=text)
                   for label, style, text in [("pythia", "--", "MG5+Pythia prior"),
                                              ("herwig", "--", "Herwig prior"),
                                              ("reference", "-", "Fixed order"),
                                              ("pythia", "-", "MaxEnt, MG5+Pythia"),
                                              ("herwig", "-", "MaxEnt, Herwig")]]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .99), ncol=3, frameon=False)
        fig.subplots_adjust(top=.87, bottom=.09, left=.09, right=.98)
        for suffix in ("pdf", "png"):
            fig.savefig(output / ("dy_portability." + suffix), dpi=180)
        plt.close(fig)
    print(f"Saved {output}. This redraw uses full-sample caches, not the compact example fits.")


if __name__ == "__main__":
    main()
