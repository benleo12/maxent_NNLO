"""Command-line access to the released moments and positive event weights."""
import argparse
from pathlib import Path

import numpy as np

from . import released
from .reweight import upgrade
from .uncertainty import histogram_bands, upgrade_with_uncertainties


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
    if args.uncertainties:
        if args.cross_section is not None:
            raise ValueError("--cross-section cannot override the correlated rates of an uncertainty ensemble")
        ensemble = upgrade_with_uncertainties(
            events, args.process, sources=args.sources, n_boot=args.bootstrap, seed=args.seed,
            born=args.born, recoil=args.recoil, mixed=args.mixed, window=args.window, nnlo_zj=args.nnlo_zj,
            progress=lambda label, result: print(f"{label}: {result.summary()}", flush=True),
        )
        result = ensemble.central
        additions = ensemble.arrays(unit_area=args.unit_area)
        for note in ensemble.notes:
            print(note)
    else:
        result = upgrade(events, moments, config)
        additions = dict(weight_upgraded=result.weights * total,
                         weight_normalization=np.asarray("unit_area" if args.unit_area else "pb"),
                         moment_names=np.asarray(list(result.chosen_moments), dtype=str),
                         moment_orders=np.asarray(list(result.chosen_moments.values()), dtype=int))
        if not args.unit_area:
            additions["cross_section_upgraded_pb"] = np.asarray(total)
    print(result.describe())
    collisions = set(events).intersection(additions)
    if collisions:
        raise ValueError(f"input already contains output fields: {sorted(collisions)}")
    saved = dict(events, **additions)
    np.savez_compressed(output, **saved)
    convention = "unit-area weights" if args.unit_area else f"cross-section weights summing to {total:.6g} pb"
    print(f"\nSaved {output}: original arrays plus weight_upgraded ({convention}).")
    if sigma.get("note") and args.cross_section is None and not args.unit_area:
        print(sigma["note"])


def histogram(args):
    """Build bands after histogramming coherent variations, for any saved observable."""
    output = Path(args.output)
    if output.suffix != ".npz" or output.resolve() == Path(args.sample).resolve():
        raise ValueError("choose a separate .npz output file")
    if output.exists() and not args.force:
        raise ValueError(f"output already exists: {output}; choose another path or use --force")
    with np.load(args.sample, allow_pickle=False) as sample:
        values = sample[args.observable]
        lo, hi = args.limits if args.limits else (values.min(), values.max())
        if not np.isfinite([lo, hi]).all() or not lo < hi or args.bins < 1:
            raise ValueError("use a finite increasing range and a positive number of bins")
        if args.log_bins and lo <= 0:
            raise ValueError("logarithmic bins need a positive lower edge")
        edges = (np.geomspace if args.log_bins else np.linspace)(lo, hi, args.bins + 1)
        hist = histogram_bands(values, edges, sample, unit_area=args.unit_area, density=True)
        hist["observable"] = np.asarray(args.observable)
        hist["normalization"] = np.asarray("unit_area" if args.unit_area or "cross_section_upgraded_pb" not in sample else "pb")
    np.savez_compressed(output, **hist)
    print(f"Saved {output}: central density and a separate band for each available source.")


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
    solve.add_argument("--uncertainties", action="store_true", help="re-solve available input variations")
    solve.add_argument("--sources", nargs="+", choices=("scale", "stat", "window", "pdf"),
                       default=["scale", "stat", "window", "pdf"], help="uncertainty sources to propagate")
    solve.add_argument("--bootstrap", type=int, default=20, help="independent-run bootstrap replicas (default: 20)")
    solve.add_argument("--seed", type=int, default=1, help="bootstrap random seed")
    solve.set_defaults(run=reweight)
    hist = commands.add_parser("histogram", help="histogram a saved observable with separate uncertainty bands")
    hist.add_argument("sample")
    hist.add_argument("observable")
    hist.add_argument("output", help="output NPZ file of bin edges and densities")
    hist.add_argument("--bins", type=int, default=30)
    hist.add_argument("--range", nargs=2, type=float, dest="limits", metavar=("LOW", "HIGH"))
    hist.add_argument("--log-bins", action="store_true")
    hist.add_argument("--unit-area", action="store_true")
    hist.add_argument("--force", action="store_true")
    hist.set_defaults(run=histogram)
    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
