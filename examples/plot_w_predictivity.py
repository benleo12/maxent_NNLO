"""Redraw the full W study from released compact numerical caches.

    python examples/plot_w_predictivity.py --output w_study_plots

This reads the full-production results; it does not refit events. For a new fit
and independent compact-sample choice, use w_constraint_workflow.py. Raw means,
separate response covariance and all twenty candidates remain in the input cache.
The old-cut map is explicitly exploratory; the 28/32 GeV cuts were prespecified.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from w_prior import TABLE, prior_asymmetry
from matplotlib.ticker import MaxNLocator

from maxent_upgrade.provenance import canonical_json, file_hash
from maxent_upgrade.study_report import STYLE


EDGES = np.array([0, 0.21, 0.42, 0.63, 0.84, 1.05, 1.37, 1.52, 1.74, 1.95, 2.18, 2.5])
MODELS = ("P", "anchor", "R", "R+H+S")
COLORS = {"P": "#87939d", "anchor": "#23658b", "R": "#cf7a2b", "R+H+S": "#bd3c3c"}
STYLES = {"P": ":", "anchor": "--", "R": "-.", "R+H+S": "-"}
LEDGER_LABELS = [f"expl_qc_{q}_{c}_pt30" for q in range(3) for c in range(3)]
CUTS = ("pt28", "pt32", "pt32hard", "pt32_given_hard", "mt60", "combined")


def band(ax, values, color, alpha=0.16):
    lo, hi = values.min(0), values.max(0)
    ax.fill_between(
        EDGES, np.r_[lo, lo[-1]], np.r_[hi, hi[-1]], step="post", color=color, alpha=alpha, lw=0
    )


def plot_asymmetry(cache, output):
    with np.load(cache / "asymmetry.npz", allow_pickle=False) as archive:
        data = dict(archive)
    meta = json.loads(str(data["metadata"]))
    model_index = {name: meta["models"].index(name) for name in MODELS}
    pairs = [tuple(pair) for pair in meta["pairs"]]
    cov = data["covariance_numerical"] + data["covariance_prior"]
    paired_cov = data["paired_covariance_numerical"] + data["paired_covariance_prior"]
    fig, axes = plt.subplots(
        3, 2, figsize=(10.2, 7.6), sharex="col", gridspec_kw={"height_ratios": [2.2, 1.0, 1.0]}
    )
    mids = (EDGES[1:] + EDGES[:-1]) / 2
    for col, cut in enumerate((28, 32)):
        ix = [meta["labels"].index(f"eta{cut}_{i:02d}") for i in range(11)]
        top, ratio, change = axes[:, col]
        ref = data["reference"][:, ix]
        referr = np.sqrt(np.maximum(np.diag(data["reference_covariance"])[ix], 0))
        if not np.all(np.isfinite(ref) & (ref > 0)):
            raise ValueError(f"Invalid NNLO asymmetry for {cut} GeV")
        band(top, ref, "#777777", alpha=0.22)
        band(ratio, ref / ref[0], "#777777", alpha=0.22)
        top.errorbar(mids, ref[0], yerr=referr, fmt="o", ms=2.2, lw=0.6, color="black", zorder=5)
        ratio.errorbar(
            mids, np.ones_like(referr), yerr=referr / ref[0], fmt="o", ms=2.2, lw=0.6, color="black", zorder=5
        )
        for name, k in model_index.items():
            if name == "P":
                raw = prior_asymmetry(cut, EDGES)
                for ax, values in [(top, raw), (ratio, raw / ref[0])]:
                    ax.stairs(values, EDGES, baseline=None, color=COLORS[name], ls=STYLES[name], lw=1.05)
                continue
            ys = data["means"][k][:, ix]
            err = np.sqrt(np.maximum(np.diag(cov[k])[ix], 0))
            for ax, values, bar in ((top, ys, err), (ratio, ys / ref, err / ref[0])):
                ax.stairs(
                    values[0],
                    EDGES,
                    color=COLORS[name],
                    ls=STYLES[name],
                    lw=1.05,
                    baseline=None,
                    zorder=3,
                )
                if name == "R+H+S":
                    band(ax, values, COLORS[name])
                    ax.errorbar(
                        mids + 0.012,
                        values[0],
                        yerr=bar,
                        fmt="none",
                        lw=0.55,
                        color=COLORS[name],
                        zorder=4,
                    )
        for first, second in (("R", "anchor"), ("R+H+S", "R")):
            values = (
                data["means"][model_index[first]][:, ix] - data["means"][model_index[second]][:, ix]
            )
            k = pairs.index((first, second))
            err = np.sqrt(np.maximum(np.diag(paired_cov[k])[ix], 0))
            band(change, values, COLORS[first], alpha=0.12)
            change.stairs(
                values[0],
                EDGES,
                color=COLORS[first],
                ls=STYLES[first],
                lw=1.05,
                baseline=None,
                zorder=3,
            )
            change.errorbar(
                mids + (0.012 if first == "R+H+S" else -0.012),
                values[0],
                yerr=err,
                fmt="none",
                lw=0.55,
                color=COLORS[first],
                zorder=4,
            )
        top.set_title(r"$p_{\mathrm{T}}^\ell>%d\,\mathrm{GeV}$" % cut, pad=5)
        ratio.axhline(1, color="#777777", lw=0.5, zorder=0)
        change.axhline(0, color="#777777", lw=0.5, zorder=0)
        for ax in (top, ratio, change):
            ax.set_xlim(0, 2.5)
            ax.margins(y=0.15)
            ax.tick_params(axis="y", pad=2)
            ax.yaxis.set_major_locator(MaxNLocator(4))
        change.tick_params(axis="x", pad=5)
        change.set_xlabel(r"$|\eta_\ell|$")
        if col == 0:
            top.set_ylabel(r"$A_\ell$")
            ratio.set_ylabel("Ratio to NNLO")
            change.set_ylabel(r"paired $\Delta A_\ell$")
    # Equal tier ranges make the threshold comparison readable without clipping.
    for row in range(3):
        limits = [ax.get_ylim() for ax in axes[row]]
        lo, hi = min(v[0] for v in limits), max(v[1] for v in limits)
        for ax in axes[row]:
            ax.set_ylim(lo, hi)
    labels = {
        "P": "LO+PS prior",
        "anchor": "MaxEnt: blended recoil",
        "R": "MaxEnt: windowed recoil",
        "R+H+S": "MaxEnt: recoil–spin moments",
    }
    handles = [
        Line2D([0], [0], color=COLORS[name], ls=STYLES[name], lw=1.1, label=label)
        for name, label in labels.items()
    ]
    handles += [
        Line2D([0], [0], color="black", marker="o", ms=2.2, lw=0, label="Independent NNLO"),
        Patch(facecolor="#777777", alpha=0.22, label="Scale variations"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.995),
        ncol=2,
        frameon=False,
        columnspacing=2.0,
        handlelength=2.5,
    )
    fig.subplots_adjust(top=0.83, bottom=0.10, left=0.10, right=0.975, hspace=0.09, wspace=0.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_caption.txt").write_text(
        "The ATLAS asymmetry measurement uses the 20 GeV selection. "
        "The paired changes show the effect of the recoil window and the spin moments. "
        "Bands span the scale variations and bars give numerical uncertainties.\n")
    fig.canvas.draw()
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(fig)


def read_metrics(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        indices = [meta["labels"].index(label) for label in LEDGER_LABELS]
        model_indices = [meta["models"].index(name) for name in ("P", "R", "R+H+S")]
        ref = z["reference"][0, indices]
        ref_cov = z["reference_covariance"][np.ix_(indices, indices)]
        cells, errors = [], []
        for k in model_indices:
            means = z["means"][k, 0, indices]
            cov = (z["covariance_numerical"][k] + z["covariance_prior"][k])[
                np.ix_(indices, indices)
            ]
            cells.append(100 * (means - ref).reshape(3, 3))
            errors.append(100 * np.sqrt(np.maximum(np.diag(cov + ref_cov), 0)).reshape(3, 3))
    return np.asarray(cells), np.asarray(errors)


def plot_old_cut_ledger(cache, output):
    data = [read_metrics(cache / f"{charge}_metrics.npz") for charge in ("WP", "WM")]
    extent = max(float(np.max(np.abs(value))) for value, _ in data)
    assert np.isfinite(extent) and extent > 0
    # Common symmetric range for both charges and both fits, with no clipped cells.
    scale = 10 ** np.floor(np.log10(extent))
    limit = np.ceil(extent / scale * 2) * scale / 2
    fig = plt.figure(figsize=(10.7, 6.0))
    axes = np.empty((2, 3), dtype=object)
    titles = ("LO+PS prior", "Windowed recoil", "With recoil–spin moments")
    for row, ((value, error), charge) in enumerate(zip(data, (r"W^{+}", r"W^{-}"))):
        for col in range(3):
            ax = fig.add_axes([0.117 + col * 0.296, 0.57 - row * 0.36, 0.253, 0.27])
            axes[row, col] = ax
            mesh = ax.pcolormesh(
                np.arange(4),
                np.arange(4),
                value[col],
                cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                edgecolors="white",
                linewidth=0.5,
                shading="flat",
            )
            for q, c in np.argwhere(np.abs(value[col]) <= error[col]):
                ax.add_patch(
                    Rectangle(
                        (c, q),
                        1,
                        1,
                        facecolor="none",
                        edgecolor="#666666",
                        hatch="///",
                        linewidth=0,
                    )
                )
            ax.set_xticks([0.5, 1.5, 2.5], labels=[r"$[0,0.6]$", r"$(0.6,0.7)$", r"$[0.7,1]$"])
            ax.set_yticks([0.5, 1.5, 2.5], labels=[r"$[0,25)$", r"$[25,50)$", r"$[50,\infty)$"])
            ax.set(xlim=(0, 3), ylim=(0, 3), title=r"$" + charge + r"$: " + titles[col])
            ax.tick_params(which="both", length=0, top=False, right=False, pad=3)
            if col == 0:
                ax.set_ylabel(r"$q_T$ interval [GeV]")
            else:
                ax.set_yticklabels([])
            if row == 1:
                ax.set_xlabel(r"$|\cos\theta^{*}|$ interval")
    cax = fig.add_axes([0.25, 0.09, 0.61, 0.028])
    bar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    bar.set_label(r"Predicted $-$ NNLO accepted fraction [percentage points]", labelpad=3)
    fig.legend(
        [Patch(facecolor="white", edgecolor="#666666", hatch="///")],
        [r"Difference within one numerical standard error"],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.96),
        frameon=False,
    )
    fig.suptitle(r"$p_{\mathrm{T}}^\ell>30\,\mathrm{GeV}$", y=1.0, fontsize=12)
    fig.canvas.draw()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(fig)


def write_acceptances(cache, output):
    """Export every candidate/scale from already normalized metric caches."""
    rows = []
    for charge in ("WP", "WM"):
        with np.load(cache / f"{charge}_metrics.npz", allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            for cut in CUTS:
                k = meta["labels"].index(cut)
                for i, model in enumerate(meta["models"]):
                    for scale, value in enumerate(z["means"][i, :, k]):
                        rows.append(
                            dict(
                                charge=charge,
                                selection=cut,
                                model=model,
                                scale_index=scale,
                                fraction=float(value),
                                central_training_error=float(
                                    np.sqrt(max(z["covariance_numerical"][i, k, k], 0))
                                ),
                                central_prior_error=float(
                                    np.sqrt(max(z["covariance_prior"][i, k, k], 0))
                                ),
                                reference_fraction=float(z["reference"][scale, k]),
                                central_reference_error=float(
                                    np.sqrt(max(z["reference_covariance"][k, k], 0))
                                ),
                            )
                        )
    with (output / "acceptances.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data/w_predictivity/full_study",
    )
    parser.add_argument("--output", type=Path, default=Path("w_study_plots"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with plt.rc_context(dict(STYLE, **{"text.usetex": False})):
        plot_asymmetry(args.data, args.output / "charge_asymmetry")
        plot_old_cut_ledger(args.data, args.output / "old_cut_ledger")
    rows = write_acceptances(args.data, args.output)
    record = dict(
        scope="full-production frozen W predictions; distinct from the compact fitting tutorial",
        input_sha256={
            name: file_hash(args.data / name)
            for name in ("asymmetry.npz", "WP_metrics.npz", "WM_metrics.npz")
        },
        prior_sha256=file_hash(TABLE),
        uncertainties="central target/refitted-prior covariance; independent reference error separate; seven scales are coherent alternatives",
        ledger="exploratory old30GeV cut, never a selection constraint",
        coordinate="charged-lepton rapidity, displayed as eta in the massless convention",
        acceptances=rows,
    )
    (args.output / "report.json").write_text(canonical_json(record) + "\n")
    print(f"Saved figures, all-candidate acceptances and provenance in {args.output}")


if __name__ == "__main__":
    main()
