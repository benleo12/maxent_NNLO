"""Released reweighting setups from the paper and the letter.

    python examples/paper_examples.py --list
    python examples/paper_examples.py wp-angle my_wp_prior.npz wp_angle.npz
    python examples/paper_examples.py zj-no-joint my_zj_prior.npz zj_separate.npz

Pass --inspect after the name to see the required arrays before running a solve.
The same diphoton setup applies to the merged and 2-to-2 priors. The same Z+jet
setup applies to priors generated with and without lepton cuts.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maxent_upgrade.cli import main as run


# name: process, additional load_moments selections, purpose
EXAMPLES = {
    "dy": ("dy", [], "Drell-Yan mass, rapidity and recoil"),
    "wp": ("wp", [], "W+ mass, rapidity and recoil"),
    "wm": ("wm", [], "W- mass, rapidity and recoil"),
    "wp-angle": ("wp", ["--recoil", "pt_w", "ctw_window"], "W+ with decay-angle moments"),
    "wm-angle": ("wm", ["--recoil", "pt_w", "ctw_window"], "W- with decay-angle moments"),
    "ggh": ("ggh", [], "Inclusive Higgs production, 54 GeV transition"),
    "aa": ("aa", [], "Diphoton mass, angle and recoil; either prior"),
    "zj": ("zj", [], "Z+jet with joint jet moments; either generation selection"),
    "zj-no-joint": ("zj", ["--mixed"], "Z+jet with separate jet moments only"),
    "zj-acoplanarity": ("zj", ["--recoil", "ptj1", "ptj2", "aco"], "Z+jet with acoplanarity moments"),
    "dy-nnlo-zj": ("dy", ["--nnlo-zj"], "Letter: Drell-Yan with NNLO Z+jet recoil input"),
    "dy-joint": ("dy", ["--mixed", "mll_x_abs_costh"], "Letter: joint mass and decay-angle moments"),
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("example", nargs="?", choices=EXAMPLES)
    parser.add_argument("prior", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--window", choices=("nominal", "down", "up"), default="nominal")
    norm = parser.add_mutually_exclusive_group()
    norm.add_argument("--unit-area", action="store_true")
    norm.add_argument("--cross-section", type=float)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.list or args.example is None:
        for name, (_, _, description) in EXAMPLES.items():
            print(f"{name:18} {description}")
        return
    process, selection, _ = EXAMPLES[args.example]
    if args.inspect:
        run(["inspect", process, *selection, "--window", args.window])
        return
    if args.prior is None:
        parser.error("provide the prior NPZ file, or use --inspect")
    argv = ["reweight", process, args.prior]
    if args.output:
        argv.append(args.output)
    argv.extend([*selection, "--window", args.window])
    if args.unit_area:
        argv.append("--unit-area")
    if args.cross_section is not None:
        argv.extend(["--cross-section", str(args.cross_section)])
    if args.force:
        argv.append("--force")
    run(argv)


if __name__ == "__main__":
    main()
