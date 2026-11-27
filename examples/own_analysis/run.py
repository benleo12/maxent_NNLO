"""Real Higgs shower -> user observable -> varied-rate fit -> HepMC -> Rivet.

This is an interoperability and normalization check, not a precision Higgs study.
All generated records, executables, fit products and audit logs go to OUTPUT.
"""

from pathlib import Path
import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pyhepmc

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from maxent_upgrade import fit_targets, released
from maxent_upgrade.hepmc import extract_events, attach_weights
from maxent_upgrade.provenance import file_hash

BR = 0.00227
PT_EDGES = np.array([0.0, 20.0, 45.0, 80.0, 150.0, 300.0, 13000.0])
SELECTION = "Inclusive gg->H->gamma gamma; fit abs(y_gamma_gamma) with linear clipped map [0,4]; no photon cuts in fit."


def higgs_photons(event):
    # Follow H and photon copies forward, avoiding unrelated hadron decays.
    todo = [p for p in event.particles if p.pid == 25]
    seen, photons = set(), {}
    while todo:
        p = todo.pop()
        if p.id in seen:
            continue
        seen.add(p.id)
        if p.pid == 22 and p.status == 1:
            photons[p.id] = p
        else:
            todo.extend(q for q in p.children if q.pid in (22, 25))
    if len(photons) != 2:
        raise ValueError(
            f"event {event.event_number}: expected exactly two Higgs-decay final photons"
        )
    return list(photons.values())


def measure(event, to_GeV):
    """A user's scalar-Higgs observable and subsequent photon selections.

    These deliberately simple photon cuts have no isolation requirement and are
    not labelled an exact ATLAS analysis. No jet or hadron-level input is needed.
    """
    photons = higgs_photons(event)
    p = (
        np.array([[q.momentum.px, q.momentum.py, q.momentum.pz, q.momentum.e] for q in photons])
        * to_GeV
    )
    pair = p.sum(axis=0)
    mass = math.sqrt(max(0.0, pair[3] ** 2 - pair[:3] @ pair[:3]))
    pt = np.hypot(p[:, 0], p[:, 1])
    eta = np.arcsinh(p[:, 2] / pt)
    rapidity = 0.5 * math.log((pair[3] + pair[2]) / (pair[3] - pair[2]))
    diphoton_pt = float(np.hypot(pair[0], pair[1]))
    fiducial = bool(
        105 < mass < 160
        and max(pt) > 0.35 * mass
        and min(pt) > 0.25 * mass
        and np.all(abs(eta) < 2.37)
        and not np.any((abs(eta) > 1.37) & (abs(eta) < 1.52))
    )
    return dict(
        h_y_abs=abs(rapidity),
        pt_gg=diphoton_pt,
        photon_fiducial=float(fiducial),
        photon_fiducial_pt45=float(fiducial and diphoton_pt > 45),
    )


def target_table():
    """Exactly two physical released NNLO moments, renamed to our event column."""
    config = {"born": {"h_y_abs": {"range": [0, 4], "map": "lin"}}, "moment_selection": False}

    def member(scale):
        moments, _ = released.load_moments(
            "ggh", born=["y_abs"], recoil=[], mixed=[], scale=scale, check_seam=False
        )
        tower = moments["born"]["y_abs"]
        return dict(
            cross_section_pb=released.cross_section("ggh", scale=scale) * BR,
            moments=[
                dict(
                    observable="h_y_abs",
                    degree=n,
                    value=float(tower["values"][n - 1]),
                    error=float(tower["errors"][n - 1]),
                    stat_error=float(tower["stat_errors"][n - 1]),
                )
                for n in (1, 2)
            ],
        )

    table = dict(config=config, selection=SELECTION, **member(None))
    labels = released.table("ggh")["scales"]
    table["variations"] = {"scale": [dict(label=labels[k], **member(k)) for k in range(1, 7)]}
    return table


def run_command(command, log, env=None):
    with open(log, "w") as output:
        result = subprocess.run(
            list(map(str, command)), stdout=output, stderr=subprocess.STDOUT, env=env
        )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}); see {log}: {shlex.join(list(map(str, command)))}"
        )


def config_flags(program, flag):
    return shlex.split(subprocess.check_output([program, flag], text=True))


def build(output, generate):
    cxx = os.environ.get("CXX", "c++")
    rivet = shutil.which("rivet-config")
    if not rivet:
        raise RuntimeError("rivet-config is required (Rivet 3 with HepMC3)")
    flags = config_flags(rivet, "--cppflags")
    libs = config_flags(rivet, "--libs")
    # A separately upgraded FastJet can leave stale -L paths in rivet-config.
    fj = shutil.which("fastjet-config")
    if fj:
        prefix = subprocess.check_output([fj, "--prefix"], text=True).strip()
        libs = ["-L" + str(Path(prefix) / "lib"), *libs]
    run_command(
        [
            cxx,
            "-std=c++17",
            "-O2",
            HERE / "rivet_runner.cc",
            "-o",
            output / "rivet_runner",
            *flags,
            *libs,
        ],
        output / "build_rivet.log",
    )
    if generate:
        pythia = shutil.which("pythia8-config")
        if not pythia:
            raise RuntimeError(
                "pythia8-config required for --generate; alternatively supply --input"
            )
        # Rivet supplies HepMC3 include/library paths already used by its reader.
        run_command(
            [
                cxx,
                "-std=c++17",
                "-O2",
                HERE / "generate_higgs.cc",
                "-o",
                output / "generate_higgs",
                *config_flags(pythia, "--cxxflags"),
                *flags,
                *config_flags(pythia, "--libs"),
                *libs,
            ],
            output / "build_pythia.log",
        )


def read_yoda(path):
    """Read only the genuine Rivet counter/histogram output used by this check."""
    results = {}
    current = None
    data = False
    for line in Path(path).read_text().splitlines():
        if line.startswith("BEGIN "):
            _, kind, name = line.split(maxsplit=2)
            current = (kind, name)
            data = False
            results[name] = []
        elif line.startswith("END "):
            current = None
        elif current and line.strip() == "---":
            data = True
        elif current and data and line.strip() and not line.startswith("#"):
            results[current[1]].append(line.split())
    return results


def direct_predictions(events, weights, mask):
    fid = events["photon_fiducial"].astype(bool) & mask
    cut = events["photon_fiducial_pt45"].astype(bool) & mask
    return {
        name: dict(
            all=float(w[mask].sum()),
            fiducial=float(w[fid].sum()),
            fiducial_pt45=float(w[cut].sum()),
            fiducial_pt=np.histogram(events["pt_gg"][fid], PT_EDGES, weights=w[fid])[0].tolist(),
        )
        for name, w in weights.items()
    }


def compare_yoda(path, expected):
    objects = read_yoda(path)
    worst = 0.0
    for weight, predictions in expected.items():
        for observable, want in predictions.items():
            key = "/MC_MAXENT_HGG/" + observable + "[" + weight + "]"
            if key not in objects:
                raise AssertionError(f"missing named-weight object {key} in {path}")
            rows = objects[key]
            if observable == "fiducial_pt":
                numeric = [row for row in rows if row[0] not in ("Total", "Underflow", "Overflow")]
                got = np.array([float(row[2]) for row in numeric])
            else:
                got = float(rows[0][0])
            want = np.asarray(want)
            error = float(np.max(np.abs(got - want)))
            worst = max(worst, error)
            # HepMC GenCrossSection text has nine significant digits; YODA is written at full precision.
            np.testing.assert_allclose(got, want, rtol=2e-8, atol=2e-12, err_msg=key)
    return worst


def check_record_preservation(source, attached, names):
    worst = 0.0
    count = 0
    with pyhepmc.open(source) as original, pyhepmc.open(attached) as output:
        for old, new in zip(original, output):
            assert old.event_number == new.event_number
            assert list(new.weights[: len(old.weights)]) == list(old.weights)
            assert new.run_info.weight_names[-len(names):] == list(names)
            assert [(p.id, p.pid, p.status) for p in old.particles] == [
                (p.id, p.pid, p.status) for p in new.particles
            ]
            a = np.array([tuple(p.momentum) for p in old.particles])
            b = np.array([tuple(p.momentum) for p in new.particles])
            np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-11)
            worst = max(worst, float(abs(a - b).max()))
            graph = lambda event: [
                (v.id, tuple(p.id for p in v.particles_in), tuple(p.id for p in v.particles_out))
                for v in event.vertices
            ]
            assert graph(old) == graph(new)
            if old.cross_section is not None:
                assert old.cross_section.xsec() == new.cross_section.xsec()
            count += 1
    return dict(
        records=count,
        maximum_four_momentum_difference_GeV=worst,
        original_weights_and_graph_preserved=True,
    )


def rates_file(path, weights, mask):
    # Only fitted streams are overridden by the deliberate negative control.
    rows = [f"{name} {w[mask].sum():.17g}" for name, w in weights.items()]
    path.write_text("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--generate", type=int, metavar="EVENTS")
    inputs.add_argument(
        "--input", type=Path, help="positive-weight inclusive H->gamma gamma HepMC3 records"
    )
    parser.add_argument("--seed", type=int, default=91357)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    build(output, bool(args.generate))
    source = args.input.resolve() if args.input else output / "prior.hepmc"
    if args.generate:
        run_command(
            [output / "generate_higgs", source, args.generate, args.seed], output / "generation.log"
        )
    events = extract_events(source, measure, selection=SELECTION)
    np.savez_compressed(output / "prior.npz", **events)
    table = target_table()
    (output / "targets.json").write_text(json.dumps(table, indent=2) + "\n")
    fit = fit_targets(events, table)
    arrays = fit.arrays()
    np.savez_compressed(output / "posterior.npz", **events, **arrays)
    weights = {"maxent_pb_per_event": arrays["weight_upgraded"]}
    weights.update(
        {
            f"maxent_scale_{i + 1:02d}_pb_per_event": row
            for i, row in enumerate(arrays["weight_scale"])
        }
    )
    report = dict(
        sample="real shower, physical released NNLO rapidity inputs; interoperability check only",
        source_sha256=file_hash(source),
        records=len(events["weight"]),
        seed=args.seed if args.generate else None,
        target_sha256=file_hash(output / "targets.json"),
        branching_fraction=BR,
        source_target_hashes={
            p.name: file_hash(p)
            for p in [
                REPO / "moments/ggh_nnlo_moments.json",
                REPO / "moments/ggh_nnlo_moments_runs.json.gz",
            ]
        },
        normalization="weights already include the H->gamma gamma BR and are pb/event",
        pT_edges_GeV=PT_EDGES.tolist(),
        checks={},
        versions=dict(
            python=sys.version.split()[0],
            numpy=np.__version__,
            pyhepmc=pyhepmc.__version__,
            rivet=subprocess.check_output(["rivet-config", "--version"], text=True).strip(),
        ),
        code_sha256={
            p.name: file_hash(p)
            for p in [
                HERE / "run.py",
                HERE / "generate_higgs.cc",
                HERE / "rivet_runner.cc",
                HERE / "MC_MAXENT_HGG.cc",
                REPO / "maxent_upgrade/hepmc.py",
            ]
        },
    )
    generator_metadata = Path(str(source) + ".json")
    if generator_metadata.is_file():
        report["generator"] = json.loads(generator_metadata.read_text())
        report["seed"] = report["generator"]["seed"]
    masks = {
        "full": np.ones(len(events["weight"]), dtype=bool),
        "fiducial": events["photon_fiducial"].astype(bool),
    }
    if not (0 < masks["fiducial"].sum() < len(events["weight"])):
        raise RuntimeError("sample must exercise both passing and failing photon selections")
    for label, mask in masks.items():
        ids = events["event_id"][mask]
        # Deliberately permute every attachment, proving association uses IDs.
        order = np.arange(len(ids))[::-1]
        selected = {name: w[mask][order] for name, w in weights.items()}
        attach_weights(
            source,
            output / (label + ".hepmc"),
            ids[order],
            selected.pop("maxent_pb_per_event"),
            expected_source_sha256=report["source_sha256"],
            variations_pb=selected,
            provenance={"target_sha256": report["target_sha256"], "selection": label},
        )
        expected = direct_predictions(events, weights, mask)
        report[label] = expected
        rates_file(output / (label + "_rates.tsv"), weights, mask)
        for mode in ("absolute", "normalized"):
            yoda = output / (label + "_" + mode + ".yoda")
            command = [output / "rivet_runner", output / (label + ".hepmc"), yoda, mode]
            # Normal mode reads the per-weight GenCrossSection from HepMC.
            run_command(command, output / (label + "_" + mode + ".log"))
            report["checks"][label + "_" + mode + "_max_abs_pb"] = compare_yoda(yoda, expected)
    # A deliberately wrong full-rate normalization on a filtered record file
    # must fail the physics normalization, and documents exactly why.
    wrong = output / "fiducial_wrong_full_rate.yoda"
    run_command(
        [
            output / "rivet_runner",
            output / "fiducial.hepmc",
            wrong,
            "normalized",
            output / "full_rates.tsv",
        ],
        output / "fiducial_wrong_full_rate.log",
    )
    wrong_objects = read_yoda(wrong)
    for name in weights:
        got = float(wrong_objects["/MC_MAXENT_HGG/all[" + name + "]"][0][0])
        full = report["full"][name]["all"]
        fid = report["fiducial"][name]["all"]
        assert abs(got / full - 1) < 2e-8 and abs(got / fid - 1) > 0.05
    report["checks"]["wrong_full_rate_on_filtered_records_detected"] = True
    report["record_preservation"] = check_record_preservation(
        source, output / "full.hepmc", weights
    )
    assert report["record_preservation"]["records"] == report["records"]
    report["checks"]["passed"] = True
    (output / "qa.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "QA": str(output / "qa.json"),
                "records": report["records"],
                "central_pb": report["full"]["maxent_pb_per_event"],
                "checks": report["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
