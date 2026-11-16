"""Plot the accuracy and numerical fluctuations of the two-jet corrections."""

from pathlib import Path
import argparse
import html
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def draw(summary, *, width=5.953, height=3.0, apply_style=True):
    if apply_style:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 8,
                "axes.labelsize": 8,
                "axes.titlesize": 8,
                "xtick.labelsize": 7.5,
                "ytick.labelsize": 7.5,
                "axes.linewidth": 0.6,
                "legend.fontsize": 7.5,
                "pdf.fonttype": 42,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.top": True,
                "ytick.right": True,
            }
        )
    fig = plt.figure(figsize=(width, height))
    axes = [
        fig.add_axes([0.10, 0.19, 0.36, 0.59]),
        fig.add_axes([0.625, 0.19, 0.35, 0.59]),
    ]
    groups = [
        (
            "MaxEnt",
            [r for r in summary["models"] if r["name"].startswith(("marg", "D4"))],
            "#b2182b",
            "o",
        ),
        (
            "Histograms, sparse cells merged",
            [r for r in summary["models"] if r["name"].startswith("merged")],
            "#2166ac",
            "D",
        ),
    ]
    handles = []
    for label, rows, colour, marker in groups:
        x = [r["constraints"] for r in rows]
        axes[0].errorbar(
            x,
            [100 * r["tv2d_supported"] for r in rows],
            yerr=[100 * r["tv2d_spread"] for r in rows],
            color=colour,
            marker=marker,
            ls="-",
            lw=0.75,
            ms=3.3,
            elinewidth=0.65,
            capsize=1.6,
        )
        axes[1].plot(
            x,
            [100 * r["correction_relative_rms_common"] for r in rows],
            color=colour,
            marker=marker,
            lw=0.75,
            ms=3.3,
        )
        handles.append(
            Line2D([], [], color=colour, marker=marker, lw=0.75, ms=3.3, label=label)
        )
    # The open red point uses separate moments; following points add joint blocks.
    row = summary["models"][0]
    for ax, value in zip(
        axes, [row["tv2d_supported"], row["correction_relative_rms_common"]]
    ):
        ax.plot(
            row["constraints"],
            100 * value,
            marker="o",
            ms=3.3,
            mfc="white",
            mec="#b2182b",
            ls="none",
        )
    axes[0].set(
        title="Joint distribution", ylabel=r"$d_{\mathrm{2D}}$ [%]", ylim=(0, 11)
    )
    axes[1].set(
        title="Numerical fluctuations", ylabel="Correction fluctuation [%]", ylim=(0, 6)
    )
    for ax in axes:
        ax.set(xlabel="Number of shape constraints", xlim=(0, 30))
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.98),
        ncol=2,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.6,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).parent / "data/zj_refinement"
    )
    parser.add_argument("--output", type=Path, default=Path("zj_refinement_plots"))
    args = parser.parse_args()
    summary = json.loads((args.results / "summary.json").read_text())
    args.output.mkdir(exist_ok=False, parents=True)
    fig = draw(summary)
    fig.savefig(args.output / "refinement.pdf")
    fig.savefig(args.output / "refinement.png", dpi=220)
    plt.close(fig)
    rows = "".join(
        f"<tr><td>{html.escape(r['name'])}</td><td>{r['constraints']}</td>"
        f"<td>{100 * r['tv2d_supported']:.2f}</td>"
        f"<td>{100 * r['correction_relative_rms_common']:.2f}</td>"
        f"<td>{100 * r['correction_relative_rms']:.2f}</td></tr>"
        for r in summary["models"]
    )
    document = (
        """<!doctype html><meta charset="utf-8"><title>Two-jet refinement</title>
<style>body{max-width:960px;margin:48px auto;padding:0 24px;font:16px/1.6 system-ui;color:#17212b}
h1{font-size:30px}p{max-width:80ch}img{width:100%;border:1px solid #dde2e6;border-radius:8px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:9px;border-bottom:1px solid #dde2e6}th:first-child,td:first-child{text-align:left}</style>
<h1>Two-jet refinement at fixed statistics</h1>
<p>Moments and histograms use the same events and fixed-order integration. Sparse histogram cells
are merged using only prior counts. Every resolution is retained.</p>
<img src="refinement.png" alt="Joint-distribution discrepancy and correction fluctuations versus number of constraints">
<p>The right panel measures the RMS relative fluctuation of the correction shape within the common
populated cells. It is not a cross-section uncertainty. Errors on cut predictions need not increase
as bins are refined, because the fitted bin probabilities are constrained.</p>
<table><tr><th>Method</th><th>Constraints</th><th>Distance [%]</th><th>Correction RMS, common cells [%]</th><th>Correction RMS, full region [%]</th></tr>"""
        + rows
        + """</table>
<p>The full-region correction diagnostic is dominated, for moments, by extrapolating resampled fits
to an extreme event absent from some resamples. Its original reference cell contains one event.
The common-cell diagnostic uses the original rule of at least 50 prior events per cell and
normalizes within that region. Both results are retained; good agreement in populated cells
does not establish reliable extrapolation into sparse tails.</p>
<p><a href="refinement.pdf">Download the figure</a></p>"""
    )
    (args.output / "index.html").write_text(document)
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
