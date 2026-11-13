"""Plot any event observable before and after reweighting, normalized to unit area.

    python examples/plot_result.py posterior.npz pT_ll --label 'Dilepton transverse momentum' --unit GeV --log-x

Requires matplotlib: pip install '.[plot]'. The input is an output of the reweight
command or custom_observables.py. A synthetic target is drawn only when the input
explicitly contains weight_target.
"""
import argparse
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maxent_upgrade import histogram_bands


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample")
    parser.add_argument("observable")
    parser.add_argument("--label")
    parser.add_argument("--unit", default="")
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--range", nargs=2, type=float, dest="limits", metavar=("LOW", "HIGH"))
    parser.add_argument("--log-x", action="store_true")
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--band", default="scale", help="uncertainty source to show if present (default: scale)")
    parser.add_argument("--output", help="PDF, SVG or PNG (default: <sample>_<observable>.pdf)")
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        parser.error("plotting requires matplotlib; install with pip install '.[plot]'")
    with np.load(args.sample, allow_pickle=False) as data:
        if args.observable not in data:
            parser.error(f"missing observable {args.observable!r}; available arrays: {data.files}")
        x = data[args.observable]
        prior_label = "Synthetic prior" if "weight_target" in data else "LO+PS prior"
        weights = {prior_label: data["weight"], "MaxEnt": data["weight_upgraded"]}
        if "weight_target" in data:
            weights["Synthetic target"] = data["weight_target"]
    if x.ndim != 1 or not np.isfinite(x).all():
        parser.error("the observable must be a finite one-dimensional array")
    if any(w.shape != x.shape or not np.isfinite(w).all() or w.sum() <= 0 for w in weights.values()):
        parser.error("weights must match the event array and have a finite positive total")
    lo, hi = args.limits if args.limits else (x.min(), x.max())
    if not lo < hi or args.bins < 1:
        parser.error("use an increasing range and a positive number of bins")
    if args.log_x and lo <= 0:
        parser.error("a logarithmic axis needs a positive lower edge; specify --range LOW HIGH")
    edges = np.geomspace(lo, hi, args.bins +
                         1) if args.log_x else np.linspace(lo, hi, args.bins + 1)
    hist = {name: np.histogram(x, edges, weights=w / w.sum())[0] / np.diff(edges)
            for name, w in weights.items()}
    with np.load(args.sample, allow_pickle=False) as data:
        bands = histogram_bands(x, edges, data, unit_area=True) if "uncertainty_sources" in data else {}
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    fig, (ax, ratio) = plt.subplots(2, 1, figsize=(6.5, 5.8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})
    styles = {prior_label: ("#8a8a8a", "--"), "MaxEnt": ("#b71727", "-"),
              "Synthetic target": ("#222222", ":")}
    reference = hist[prior_label]
    if args.band + "_low" in bands:
        low, high = bands[args.band + "_low"], bands[args.band + "_high"]
        ax.stairs(high, edges, baseline=low, fill=True, color="#b71727", alpha=0.18,
                  label=f"MaxEnt {args.band} uncertainty", linewidth=0)
        ratio.stairs(np.divide(high, reference, out=np.full_like(high, np.nan), where=reference > 0), edges,
                     baseline=np.divide(low, reference, out=np.full_like(low, np.nan), where=reference > 0),
                     fill=True, color="#b71727", alpha=0.18, linewidth=0)
    for name, values in hist.items():
        color, line = styles[name]
        ax.stairs(values, edges, color=color, linestyle=line, linewidth=1.8, label=name)
        if name != prior_label:
            divided = np.divide(values, reference, out=np.full_like(
                values, np.nan), where=reference > 0)
            ratio.stairs(divided, edges, color=color, linestyle=line, linewidth=1.5, baseline=None)
    ratio.axhline(1, color="#8a8a8a", linewidth=0.8)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, borderaxespad=0)
    ax.set_ylabel("Normalized density" + (f" [1/{args.unit}]" if args.unit else ""))
    ratio.set_ylabel("Ratio to prior")
    label = args.label or args.observable
    ratio.set_xlabel(label + (f" [{args.unit}]" if args.unit else ""))
    if args.log_x:
        ratio.set_xscale("log")
    if args.log_y:
        ax.set_yscale("log")
    output = args.output or str(Path(args.sample).with_suffix("")) + f"_{args.observable}.pdf"
    fig.savefig(output)
    plt.close(fig)
    print(f"Saved {output}. Shapes use the full sample's normalization, including events outside the displayed range.")


if __name__ == "__main__":
    main()
