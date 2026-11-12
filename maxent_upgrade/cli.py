"""Command-line access to the released moments and positive event weights."""
import argparse
from pathlib import Path

import numpy as np

from . import released
from .reweight import upgrade


def _selection(parser):
    parser.add_argument("--born", nargs="*",
                        help="Born constraint keys; an empty list removes them")
    parser.add_argument("--recoil", nargs="*",
                        help="window constraint keys, including any added observables")
    parser.add_argument("--mixed", nargs="*",
                        help="joint constraint keys; an empty list removes them")
    parser.add_argument("--window", choices=("nominal", "down", "up"), default="nominal")
    parser.add_argument("--nnlo-zj", action="store_true",
                        help="use the letter's NNLO Z+jet recoil moments for dy")


def _load(args):
    return released.load_moments(
        args.process, born=args.born, recoil=args.recoil, mixed=args.mixed,
        window=args.window, nnlo_zj=args.nnlo_zj,
    )


def inspect(args):
    """Show the actual selected configuration, including transformed map intervals."""
    moments, config = _load(args)
    table = released.table(args.process)
    print(f"{table['process']} | {table['order']} input")
    print(f"Selection: {table['fiducial']}")
    print("Lepton definitions, isolation and generation settings: moments/README.md")
    print("\nRequired event arrays (one entry per event):")
    print("  weight: strictly positive prior weight, with any common normalization")
    for role in ("born", "recoil"):
        for name, spec in config.get(role, {}).items():
            target = moments[role][name]
            values = target["values"] if role == "born" else target["window_values"]
            print(
                f"  {name}: {role}, map {spec['map']}, range {spec['range']}, {len(values)} available moments")
            if "profile" in spec:
                print(f"    window {spec['profile']}, transition {spec.get('transition', 'blend')}")
    for name, spec in config.get("mixed", {}).items():
        print(
            f"  joint {name}: {spec['observables']}, {len(moments['mixed'][name]['values'])} moments")
    print("\nThe solver selects the resolved moment orders for your prior.")
    sigma = released.sigma_fid(args.process)
    print(f"Target cross section: {sigma['value']:.6g} pb")
    if sigma.get("note"):
        print(sigma["note"])


def reweight(args):
    """Load event arrays, solve once, and preserve their order in the output."""
    prior = Path(args.prior)
    output = Path(args.output) if args.output else prior.with_name(prior.stem + "_upgraded.npz")
    if output.suffix != ".npz":
        raise ValueError("the output filename must end in .npz")
    if output.resolve() == prior.resolve():
        raise ValueError("choose a different output path so the original sample is preserved")
    if output.exists() and not args.force:
        raise ValueError(f"output already exists: {output}; choose another path or use --force")
    with np.load(prior, allow_pickle=False) as data:
        events = dict(data)
    if "weight_upgraded" in events or "cross_section_upgraded_pb" in events:
        raise ValueError("the input already contains upgrade output; use the original prior file")
    moments, config = _load(args)
    sigma = released.sigma_fid(args.process)
    total = 1.0 if args.unit_area else sigma["value"] if args.cross_section is None else args.cross_section
    if not np.isfinite(total) or total <= 0:
        raise ValueError("the target cross section must be a finite positive number in pb")
    print(f"Selection required: {released.table(args.process)['fiducial']}")
    print("The package applies no cuts. Input events must already satisfy this selection.\n")
    result = upgrade(events, moments, config)
    print(result.describe())
    saved = dict(events, weight_upgraded=result.weights * total)
    if not args.unit_area:
        saved["cross_section_upgraded_pb"] = np.asarray(total)
    np.savez_compressed(output, **saved)
    convention = "unit-area weights" if args.unit_area else f"cross-section weights summing to {total:.6g} pb"
    print(f"\nSaved {output}: original arrays plus weight_upgraded ({convention}).")
    if sigma.get("note") and args.cross_section is None and not args.unit_area:
        print(sigma["note"])


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="maxent-upgrade", description="Positive event weights from fixed-order moments."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser(
        "inspect", help="show the required arrays, selection and selected constraints")
    show.add_argument("process", help="dy, wp, wm, ggh, aa, zj, or a released-format JSON table")
    _selection(show)
    show.set_defaults(run=inspect)
    solve = commands.add_parser(
        "reweight", help="reweight a prior NPZ file using a released moment table")
    solve.add_argument("process", help="dy, wp, wm, ggh, aa, zj, or a released-format JSON table")
    solve.add_argument(
        "prior", help="NPZ file of event arrays inside the table's fiducial selection")
    solve.add_argument("output", nargs="?", help="output NPZ file (default: <prior>_upgraded.npz)")
    _selection(solve)
    norm = solve.add_mutually_exclusive_group()
    norm.add_argument("--cross-section", type=float, metavar="PB",
                      help="override the total cross section in pb")
    norm.add_argument("--unit-area", action="store_true", help="save weights summing to one")
    solve.add_argument("--force", action="store_true", help="replace an existing output file")
    solve.set_defaults(run=reweight)
    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
