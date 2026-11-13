"""Reproduce the paper's binned two-jet correlation check, without event files.

    python examples/plot_joint_validation.py

The supplied bins are independent validation histograms, not inputs to the fit.
See data/README.md for their selection, normalization and integration provenance.
For a new process, histogram your separate/joint posterior weights with
numpy.histogram2d and compare with the same bins in your fixed-order calculation.
"""
from pathlib import Path

import numpy as np


def main():
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Polygon, Rectangle
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    path = Path(__file__).resolve().parent / "data" / "zj_joint_validation.npz"
    with np.load(path) as data:
        data = dict(data)
    edges = data["edges"]
    target = data["fo_per_run_fb"][..., 0].mean(axis=0)
    target /= target.sum()
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True,
                             constrained_layout=True)
    norm = TwoSlopeNorm(vmin=0.6, vcenter=1, vmax=1.4)
    for ax, key, title in zip(axes, ("marg", "joint"), ("Separate jet moments", "With joint jet moments")):
        raw = data[key]
        shape = raw / raw.sum()
        neff = np.divide(raw**2, data[key + "_w2"], out=np.zeros_like(raw), where=raw > 0)
        ratio = np.divide(shape, target, out=np.full_like(raw, np.nan), where=target > 0)
        sparse = (neff < 10) & (target > 0)
        ratio[sparse] = np.nan
        mesh = ax.pcolormesh(edges, edges, ratio.T, norm=norm, cmap="RdBu_r")
        clip = Polygon([(edges[0], edges[0]), (edges[-1], edges[0]), (edges[-1], edges[-1])],
                       transform=ax.transData)
        mesh.set_clip_path(clip)
        for i, j in zip(*np.where(sparse)):
            cell = Rectangle((edges[i], edges[j]), edges[i + 1] - edges[i], edges[j + 1] - edges[j],
                             facecolor=".94", edgecolor=".5", hatch="////", linewidth=0)
            cell.set_clip_path(clip)
            ax.add_patch(cell)
        ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], color=".5", lw=0.8)
        ax.set(xscale="log", yscale="log", xlim=(edges[0], edges[-1]), ylim=(edges[0], edges[-1]),
               xlabel=r"$p_T^{j_1}$ [GeV]", title=title, aspect="equal")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator([60, 100, 200, 400, 1000]))
            axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            axis.set_minor_formatter(NullFormatter())
        print(f"{title}: half the summed absolute bin-probability difference = {np.abs(shape - target).sum() / 2:.4f}")
    axes[0].set_ylabel(r"$p_T^{j_2}$ [GeV]")
    fig.colorbar(mesh, ax=axes, orientation="horizontal", location="top", shrink=0.65,
                 label="MaxEnt / fixed order", extend="both")
    fig.savefig("zj_joint_validation.pdf")
    plt.close(fig)
    print("Saved zj_joint_validation.pdf. Unit area in the displayed region; hatching marks fewer than 10 effective events.")


if __name__ == "__main__":
    main()
