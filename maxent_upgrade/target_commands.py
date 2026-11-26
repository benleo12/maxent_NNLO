"""Command-line input and output for user-supplied fixed-order targets."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

import numpy as np

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


@contextmanager
def _new_output(path, suffix):
    path = Path(path)
    if path.suffix != suffix:
        raise ValueError(f"output file must end in {suffix}")
    if path.exists():
        raise ValueError(f"output already exists: {path}; choose a new filename")
    with tempfile.NamedTemporaryFile(dir=path.parent) as temp:
        yield temp
        temp.flush()
        os.link(temp.name, path)


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
    with np.load(args.prior, allow_pickle=False) as data:
        events = dict(data)
    if "weight_upgraded" in events:
        raise ValueError("the prior already contains fitted weights; supply your original prior")
    if "selection" in table:
        print(f"Input fiducial selection: {table['selection']}")
    with _new_output(args.output, ".npz") as f:
        fit = fit_targets(events, table)
        fit.provenance["input_file"] = dict(name=Path(args.prior).name, sha256=file_hash(args.prior))
        fit.provenance["target_file"] = dict(name=Path(args.targets).name, sha256=file_hash(args.targets))
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


def add_target_commands(commands):
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
    book = commands.add_parser(
        "moments", help="compute named Chebyshev targets from your fixed-order integration records")
    book.add_argument("config", help="JSON defining each observable ID, map, interval and any recoil window")
    book.add_argument("fixed_order", help="NPZ with the observable columns, signed weight, optional weight_scales and scale_labels")
    book.add_argument("output", help="new target JSON for the fit command")
    book.add_argument("--orders", type=int, help="highest degree to book in each one-dimensional tower")
    book.add_argument("--cross-section", metavar="PB", type=float, help="target fiducial cross section in pb (default: unit-area weights)")
    book.add_argument("--selection", metavar="TEXT", help="record the common fiducial cuts and observable convention")
    book.set_defaults(run=_book)
