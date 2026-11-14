"""Redraw the W- joint-moment redistribution map from its compact cache."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_map(cache, output):
    """Display the same-event J2/A change; there is no independent 2D reference."""
    with np.load(cache, allow_pickle=False) as source:
        data = dict(source)
    output = Path(output)
    correction = 100 * (data["J2_over_A"] - 1)
    valid = data["display_mask"]
    limit = max(.5, float(np.ceil(np.max(np.abs(correction[valid])) * 2) / 2))
    with plt.rc_context({"font.family": "serif", "font.size": 9, "mathtext.fontset": "stix",
                         "axes.labelsize": 10, "axes.titlesize": 9,
                         "xtick.direction": "in", "ytick.direction": "in"}):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        ax.set_facecolor("#ececec")
        mesh = ax.pcolormesh(data["y_edges"], data["c_edges"],
                             np.ma.array(correction.T, mask=~valid.T), cmap="RdBu_r",
                             vmin=-limit, vmax=limit, rasterized=True)
        for edge in [-.7, -.6, .6, .7]:
            ax.axhline(edge, color="black", ls=":" if abs(edge) == .7 else "--", lw=.55, alpha=.65)
        ax.set_xlabel(r"$|y_W|$")
        ax.set_ylabel(r"Signed $\cos\theta^*_{\rm CS}$")
        ax.set_title(r"$W^-:$ redistribution from rapidity-angle moments")
        colorbar = fig.colorbar(mesh, ax=ax, pad=.025)
        colorbar.set_label(r"$100\,[P_{J2}/P_A-1]$ [%]")
        fig.text(.5, .015, "Same events; grey cells have fewer than 100 raw or effective events",
                 ha="center", fontsize=7)
        fig.tight_layout(rect=(0, .035, 1, 1))
        for suffix in ["pdf", "png"]:
            fig.savefig(output / ("wm_joint_redistribution." + suffix), dpi=220, bbox_inches="tight")
        plt.close(fig)
    metadata = json.loads(str(data["metadata"]))
    caption = ("Redistribution in the W- sample when the six degree-two rapidity-angle moments are added "
               "to the production and angular constraints. The ratio uses the same events with the frozen J2 and A weights. "
               "Grey cells have fewer than 100 raw or A-weighted effective events. Dashed and dotted lines mark the "
               "flat-window and zero-window boundaries at |cos(theta*)|=0.6 and 0.7. "
               "This is a correction map, not closure against an independent two-dimensional NNLO prediction.")
    (output / "wm_joint_redistribution_caption.txt").write_text(caption + "\n")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, nargs="?",
                        default=Path(__file__).resolve().parent / "data/w_joint/wm_joint_redistribution.npz")
    parser.add_argument("--output", type=Path, default=Path("w_joint_map_plots"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    plot_map(args.cache, args.output)


if __name__ == "__main__":
    main()
