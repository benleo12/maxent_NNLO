"""Command-line input and output for user-supplied fixed-order targets."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from ._files import _new_output
from .provenance import file_hash
from .targets import book_targets, fit_targets


def _load_json(path):
    def unique(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON field {key!r}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def _book(args):
    config = _load_json(args.config)
    with np.load(args.fixed_order, allow_pickle=False) as data:
        targets = book_targets(data, config, orders=args.orders, cross_section_pb=args.cross_section,
                               selection=args.selection)
    with _new_output(args.output, ".json") as f:
        f.write((json.dumps(targets, indent=2, allow_nan=False) + "\n").encode())
    print(f"Saved {len(targets['moments'])} named Chebyshev moments to {args.output}.")


def _fit(args):
    table = _load_json(args.targets)

    def build(events):
        fit = fit_targets(events, table)
        fit.provenance["target_file"] = dict(name=Path(args.targets).name, sha256=file_hash(args.targets))
        return fit

    if "selection" in table:
        print(f"Input fiducial selection: {table['selection']}")
    _fit_sample(args, build)


def _fit_sample(args, build):
    with np.load(args.prior, allow_pickle=False) as data:
        events = dict(data)
    if "weight_upgraded" in events:
        raise ValueError("the prior already contains fitted weights; supply your original prior")
    with _new_output(args.output, ".npz") as f:
        fit = build(events)
        fit.provenance["input_file"] = dict(name=Path(args.prior).name, sha256=file_hash(args.prior))
        arrays = fit.arrays()
        collisions = set(events).intersection(arrays)
        if collisions:
            raise ValueError(f"input already contains output fields: {sorted(collisions)}")
        np.savez_compressed(f, **events, **arrays)
    print(fit.central.describe(cross_section_pb=None if fit.unit_area_only else fit.cross_section))
    normalization = "one" if fit.unit_area_only else f"{fit.cross_section:g} pb"
    print(f"Saved {args.output}: weight_upgraded sums to {normalization}.")
    for source, weights in fit.weights.items():
        print(f"  weight_{source}: {len(weights)} variations, labels in labels_{source}")


def _match(args):
    from . import released
    from .reweight import _check_event_columns
    from .uncertainty import upgrade_with_uncertainties

    if args.jobs < 1 or args.seed < 0:
        raise ValueError("use a positive number of jobs and a nonnegative replica seed")
    if args.replicas < 0 or args.replicas == 1:
        raise ValueError("request zero or at least two integration replicas")
    if args.setup in released.PROCESSES:
        table = released.table(args.setup)
        print(f"{table['process']} | released {table['order']} input. Selection: {table['fiducial']}")
        available = table["default_constraints"]["born"] + table["default_constraints"]["recoil"]
        limits = {}
        for name, degree in args.max_degree:
            if name not in available or name in limits:
                raise ValueError(f"--max-degree requires a distinct released observable from {available}")
            limits[name] = int(degree)
            if not 1 <= limits[name] <= table["towers"][name]["n_orders"]:
                raise ValueError(f"degree for {name} must be between 1 and {table['towers'][name]['n_orders']}")
        sources = ["scale", "stat"] if args.replicas else ["scale"]
        _fit_sample(args, lambda events: upgrade_with_uncertainties(
            events, args.setup, sources=sources, n_boot=args.replicas, seed=args.seed, check_support=True,
            max_orders=limits or None))
        return
    if args.max_degree:
        raise ValueError("for a new integration specify degrees in the calculation file")
    from .nnlojet import calculate

    setup_path = Path(args.setup)
    if not setup_path.is_file():
        raise ValueError(f"give a calculation JSON or a released setup: {', '.join(released.PROCESSES)}")
    setup = _load_json(setup_path)
    if not isinstance(setup, dict) or "runcard" not in setup:
        raise ValueError("a new NNLOJET setup needs a runcard; for a table of moment values use fit")
    folder = args.work_dir or str(setup_path.with_suffix("")) + "_runs"

    def build(events):
        if not isinstance(setup.get("axes"), dict) or not isinstance(setup.get("config"), dict):
            raise ValueError("the calculation needs named axes and an observable config")
        wkey = setup["config"].get("weight_key", "weight")
        _check_event_columns(events, setup["axes"], wkey)
        if "event_id" in events and np.shape(events["event_id"]) != np.shape(events[wkey]):
            raise ValueError("supply one event ID per prior weight")
        targets = calculate(setup_path, setup, binary=args.nnlojet, directory=folder,
                            workers=args.jobs, replicas=args.replicas, seed=args.seed)
        print(f"Reusable fixed-order moments: {targets}", flush=True)
        fit = fit_targets(events, _load_json(targets))
        fit.provenance["target_file"] = dict(name=targets.name, sha256=file_hash(targets))
        fit.provenance["nnlojet_calculation"] = file_hash(targets.parent / "calculation.json")
        return fit

    _fit_sample(args, build)


def _merge(args):
    from .booking import merge_bookings
    with _new_output(args.output, ".json") as f:
        table = merge_bookings(args.jobs).targets(replicas=args.replicas, seed=args.seed)
        f.write((json.dumps(table, indent=2, allow_nan=False) + "\n").encode())
    print(f"Saved {len(table['moments'])} moments and their integration covariance to {args.output}.")


def _check(args):
    with np.load(args.sample, allow_pickle=False) as sample:
        if "constraint_checks_json" not in sample:
            raise ValueError("this file has no stored per-constraint checks; run fit again with your original prior and targets")
        checks = json.loads(str(sample["constraint_checks_json"]))
    if args.selection:
        if args.source is not None or args.member is not None:
            raise ValueError("the choice of moments is made on the central sample")
        if "prefit" not in checks:
            raise ValueError("this file predates saved selection checks; refit the original sample")
        print("Observable and degree | Prior | Fixed order | Input error | SNR | Used")
        for row in checks["prefit"]:
            values = [row[key] for key in ("prior", "target", "error", "snr")]
            fields = ["-" if val is None else f"{val:.6g}" for val in values]
            if row["snr"] is None and row["prior"] is not None and row["error"] == 0:
                fields[-1] = "inf"
            used = "yes" if row["retained"] else "no"
            print(f"{row['observable']} {row['degree']} | " + " | ".join(fields) + f" | {used} ({row['rule']})")
        return
    if args.source is None:
        if args.member is not None:
            raise ValueError("--member requires --source")
        title, rows, support = "central", checks["central"], checks["support"]
    else:
        members = checks["variations"].get(args.source)
        if members is None:
            raise ValueError(f"unknown source {args.source!r}; available: {list(checks['variations'])}")
        choices = {m["label"]: m for m in members}
        if args.member not in choices:
            raise ValueError(f"choose --member for {args.source}: {list(choices)}")
        item = choices[args.member]
        title, rows, support = f"{args.source} / {args.member}", item["constraints"], item["support"]
    print(f"{title} | hard targets: {support}")
    identifiers = []
    for row in rows:
        kind, degree = row["kind"], row["degree"]
        if degree is None:
            name = row["name"] if kind == "normalization" else f"{row['observable']} {kind}"
        else:
            tag = ",".join(map(str, degree)) if isinstance(degree, list) else str(degree)
            polynomial = f"T[{tag}]"
            if kind in ("windowed", "windowed joint"):
                polynomial = f"<w {polynomial}>"
            elif kind == "blend":
                polynomial += " (blend)"
            name = f"{row['observable']} {polynomial}"
        identifiers.append(name)
    width = max(10, max(map(len, identifiers)))
    show_mode = any(row["mode"] == "soft" for row in rows)
    mode_column = " Mode" if show_mode else ""
    print(f"{'Constraint':<{width}}{mode_column}  {'Prior':>12} {'Target':>12} {'Achieved':>12} {'Residual':>12} {'Input error':>12}")
    for name, row in zip(identifiers, rows):
        error = "-" if row["input_error"] is None else f"{row['input_error']:.6g}"
        mode = f" {row['mode']:4}" if show_mode else ""
        print(f"{name:<{width}}{mode}  {row['prior']:12.6g} {row['target']:12.6g} {row['achieved']:12.6g} "
              f"{row['residual']:+12.4g} {error:>12}")
    if any(row["kind"] in ("windowed", "windowed joint", "blend") for row in rows):
        print("Window moments are fiducial averages. Error propagation holds the prior fixed.")
        if any("integration_error" in row for row in rows):
            print("Rows with a supplied moment-rate covariance use it and their coherent scale choices.")
        if any("integration_error" not in row and row["kind"] in ("windowed", "windowed joint", "blend") for row in rows):
            print("Rows with marginal errors alone combine them in quadrature.")


def add_target_commands(commands):
    match = commands.add_parser("match", help="match your events using a released or new NNLOJET calculation")
    match.add_argument("prior", help="positive-weight shower sample")
    match.add_argument("setup", help="released setup ID or JSON specifying a new NNLOJET calculation")
    match.add_argument("output", help="new NPZ with all events and their fitted weights")
    match.add_argument("--nnlojet", default=os.environ.get("NNLOJET", "NNLOJET"),
                       help="NNLOJET executable with the moment hook (new calculations)")
    match.add_argument("--max-degree", nargs=2, metavar=("OBS", "N"), action="append", default=[],
                       help="highest degree to consider for a named released observable (may be repeated)")
    match.add_argument("--work-dir", help="store and resume independent runs here")
    match.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                       help="concurrent NNLOJET processes (default: up to four)")
    match.add_argument("--replicas", type=int, default=0, help="number of joint integration-error refits")
    match.add_argument("--seed", type=int, default=1, help="random seed for the integration-error replicas")
    match.set_defaults(run=_match)
    merge = commands.add_parser("merge-moments", help="combine independent integration bookings into a target table")
    merge.add_argument("jobs", nargs="+", help="job summary NPZ files")
    merge.add_argument("-o", "--output", required=True, help="new target JSON")
    merge.add_argument("--replicas", type=int, default=0, help="number of joint integration-error replicas")
    merge.add_argument("--seed", type=int, default=1)
    merge.set_defaults(run=_merge)
    fit = commands.add_parser(
        "fit", help="reweight your LO+PS prior using named Chebyshev moment targets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Supply a positive prior and the Chebyshev moments from your fixed-order calculation.
Each row identifies the observable (the prior column) and its polynomial degree.
The config gives its map, range and any window, shared by central and varied rows.

Example targets.json:
  {"config": {"born": {"y_abs": {"range": [0, 2.4], "map": "lin"}}},
   "moments": [{"observable": "y_abs", "degree": 1, "value": -0.12, "error": 0.004}],
   "cross_section_pb": 12.4}

Supply the degrees you measured (e.g. 2 and 4 alone). Give errors for automatic order
selection, or set "moment_selection":false in config to fit them all.
The input events must satisfy the same fiducial selection. Use T_n(cos θ)=cos(nθ)
on the mapped interval, clipping at its endpoints. Each supplied variation is
refitted at the central orders. Its label identifies its row of output weights.""")
    fit.add_argument("prior", help="NPZ with positive weight, named observable columns and optional event IDs")
    fit.add_argument("targets", help="JSON with config and identified moment rows; add a rate in pb for absolute weights")
    fit.add_argument("output", help="new NPZ with the input event arrays and aligned replacement weights")
    fit.set_defaults(run=_fit)
    check = commands.add_parser("check", help="show the per-constraint checks saved by fit")
    check.add_argument("sample", help="upgraded NPZ from the fit command")
    check.add_argument("--source", help="uncertainty source (default: central)")
    check.add_argument("--member", help="exact member label within that source")
    check.add_argument("--selection", action="store_true", help="show the central prior comparison for every supplied degree")
    check.set_defaults(run=_check)
    book = commands.add_parser(
        "moments", help="compute named Chebyshev targets from your fixed-order integration records")
    book.add_argument("config", help="JSON defining each observable ID, map, interval and any recoil window")
    book.add_argument("fixed_order", help="NPZ with the observable columns, signed weight, optional weight_scales and scale_labels")
    book.add_argument("output", help="new target JSON for the fit command")
    book.add_argument("--orders", type=int, help="highest degree to book in each one-dimensional tower")
    book.add_argument("--cross-section", metavar="PB", type=float, help="target fiducial cross section in pb (default: unit-area weights)")
    book.add_argument("--selection", metavar="TEXT", help="record the common fiducial cuts and observable convention")
    book.set_defaults(run=_book)
