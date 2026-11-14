"""A complete, small training → freeze → independent prediction tutorial.

    python examples/constraint_workflow.py output/constraint_tutorial

This controlled Born-decay illustration is not an NNLO calculation or a shower
sample. The real W example uses the same public interfaces with booked QCD
integrals. No external inputs, network access or process registration are needed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from maxent_upgrade import (ConstraintBlock, NumericalInput, bin_diagnostics, fit_constraints, load_frozen,
                            plan_observable, validate_prediction)
from maxent_upgrade.provenance import canonical_json, file_hash
from maxent_upgrade.study_report import write_study_report


LABELS = ("rate", "y2", "c2", "yc", "yc3", "y3c")
SCALES = ("central", "shape_down", "shape_up")
EDGES = np.linspace(-3, 3, 13)
MODEL_BLOCKS = {"Prior": (), "Separate moments": ("production", "angle"),
                "Joint moments": ("production", "angle", "rapidity_angle")}
DEFINITION = dict(observable="eta_l = y + atanh(c)", lepton_pt_cut_GeV=27., mass_GeV=80.,
                  edges=EDGES.tolist(), bin_contents="integral in pb", convention="massless Born decay",
                  density_label=r"$d\sigma/d\eta_\ell$ [pb]")


def columns(y, c):
    return np.column_stack([np.ones(len(y)), y**2, c**2, y * c, y * c**3, y**3 * c])


def held_out(y, c):
    eta = y + np.arctanh(c)
    accept = 40 * np.sqrt(1 - c**2) > 27
    hist = np.column_stack([(eta >= lo) & (eta < hi) & accept for lo, hi in zip(EDGES[:-1], EDGES[1:])])
    return hist.astype(float)


def target(y, c):
    # Positive spin-1-like angular density with a rapidity-dependent odd term.
    # The three columns are declared coherent illustration variations, not a
    # renormalization/factorization-scale calculation.
    return np.column_stack([100 * np.exp(-y**2 / (1.7 * (1 + delta))) *
                            (1 + c**2 + .75 * (1 + .2 * delta) * y * c)
                            for delta in (0., -.04, .04)])


def integration(role):
    labels = LABELS if role == "training" else ("rate",) + tuple(f"eta_{i}" for i in range(len(EDGES) - 1))
    seeds = np.arange(64) + (910000 if role == "training" else 920000)
    runs = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        y, c = rng.uniform(-2, 2, 512), rng.uniform(-.9, .9, 512)
        features = columns(y, c) if role == "training" else np.column_stack([np.ones(len(y)), held_out(y, c)])
        runs.append(features.T @ target(y, c) / len(y))
    runs = np.asarray(runs)[None, ...]
    data = NumericalInput.from_runs(labels, runs, seeds=seeds[None, :], channels=("illustration",),
                                    sample_id=f"controlled-{role}-v1", role=role, scale_names=SCALES,
                                    metadata=dict(description="Controlled Born-decay illustration; not fixed-order QCD",
                                                  points_per_run=512, integration_measure="uniform y,c average",
                                                  coherent_variations="illustrative production-shape variations"))
    return data, runs, seeds[None, :]


def make_blocks(events, data):
    y, c = events["y"], events["c"]
    definitions = {
        "production": (y[:, None]**2, ("y2",), "production rapidity width"),
        "angle": (c[:, None]**2, ("c2",), "polar-angle even component"),
        "rapidity_angle": ((y * c)[:, None], ("yc",), "rapidity dependence of the angular odd component"),
        "diagnostic": (columns(y, c)[:, 3:], LABELS[3:], "supplied rapidity-angle correlations"),
    }
    return {name: ConstraintBlock.from_input(name, values, data, labels=labels,
                                             metadata=dict(definition=meaning, y_range=[-2, 2], c_range=[-.9, .9]))
            for name, (values, labels, meaning) in definitions.items()}


def train(directory):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "selection.json").exists():
        raise FileExistsError("this directory already contains a frozen decision; choose a new output directory")
    rng = np.random.default_rng(731131)
    y, c = rng.uniform(-2, 2, 6000), rng.uniform(-.9, .9, 6000)
    events = dict(y=y, c=c, event_id=np.arange(len(y)), weight=np.exp(-y**2 / 2.2) * (1 + c**2))
    np.savez_compressed(directory / "prior.npz", **events)
    data, runs, seeds = integration("training")
    np.savez_compressed(directory / "training.npz", mean=data.mean, covariance=data.covariance,
                        runs=runs, seeds=seeds, metadata=np.array(canonical_json(data.identity)))
    blocks = make_blocks(events, data)
    fits, diagnostics = {}, {}
    for name, names in MODEL_BLOCKS.items():
        fits[name] = fit_constraints(events["weight"], data, [blocks[k] for k in names],
                                     prior_id="controlled-prior-v1", event_ids=events["event_id"])
        diagnostics[name] = fits[name].diagnose(blocks["diagnostic"])
    selected = "Joint moments" if diagnostics["Separate moments"]["test"]["resolved"] else "Separate moments"
    selection = dict(selected=selected, anchor="Separate moments", significance=.01, rank_cutoff=1e-6,
                     rule="Add the declared yc block if the marginal fit leaves a resolved diagnostic at every coherent variation. Keep and report every candidate.",
                     candidates={k: v["test"] for k, v in diagnostics.items()},
                     validation_opened=False,
                     limitation="A remaining discrepancy is reported, not removed by adding validation cuts as constraints.")
    h = held_out(y, c)
    labels = tuple(f"eta_{i}" for i in range(h.shape[1]))
    plan = {"eta": plan_observable(h, labels=labels, definition=DEFINITION)}
    weights = dict(event_id=events["event_id"], original_weight=events["weight"])
    for i, (name, fit) in enumerate(fits.items()):
        fit.freeze(directory / f"candidate_{i}.npz", selection=dict(selection, candidate=name), validation_plan=plan)
        weights[f"candidate_{i}_weights_pb"] = fit.scale_weights * data.mean[0, :, None]
    weights["metadata"] = np.array(canonical_json(dict(models=list(fits), scale_names=SCALES,
                                                       units="absolute pb per event; sum directly")))
    np.savez_compressed(directory / "weights.npz", **weights)
    (directory / "selection.json").write_text(canonical_json(selection) + "\n")
    print(f"Frozen {len(fits)} candidates. Training selected: {selected}.")
    print(f"Absolute event weights: {directory / 'weights.npz'}")


def validate(directory):
    selection = json.loads((directory / "selection.json").read_text())
    with np.load(directory / "prior.npz", allow_pickle=False) as z:
        events = {k: z[k] for k in z.files}
    with np.load(directory / "training.npz", allow_pickle=False) as z:
        identity = json.loads(str(z["metadata"]))
        data = NumericalInput(identity["labels"], z["mean"], z["covariance"], identity["sample_id"],
                              role="training", scale_names=identity["scale_names"],
                              metadata=identity["metadata"], run_ids=identity["run_ids"])
    blocks, fits, diagnostics = make_blocks(events, data), {}, {}
    # Restore and verify every candidate before evaluating any reference.
    for i, (name, names) in enumerate(MODEL_BLOCKS.items()):
        fits[name] = load_frozen(directory / f"candidate_{i}.npz", events["weight"], data,
                                 [blocks[k] for k in names], prior_id="controlled-prior-v1",
                                 event_ids=events["event_id"])
        diagnostics[name] = {"rapidity–angle": fits[name].diagnose(blocks["diagnostic"])}
    reference, runs, seeds = integration("validation")
    np.savez_compressed(directory / "validation.npz", mean=reference.mean, covariance=reference.covariance,
                        runs=runs, seeds=seeds, metadata=np.array(canonical_json(reference.identity)))
    h = held_out(events["y"], events["c"])
    labels = tuple(f"eta_{i}" for i in range(h.shape[1]))
    results = {}
    for name, fit in fits.items():
        prediction = fit.predict(h, labels=labels, definition=DEFINITION, validation_key="eta")
        results[name] = validate_prediction(prediction, reference, independent_of=[data.sample_id], definition=DEFINITION)
    report = write_study_report(directory / "report", title="Rapidity–angle correlations: a controlled example",
                                models=fits, diagnostics=diagnostics, validations=results,
                                edges=EDGES, xlabel=r"Lepton pseudorapidity $\eta_\ell$ after $p_{T,\ell}>27$ GeV",
                                selected=selection["selected"], anchor=selection["anchor"],
                                provenance=dict(example="controlled Born illustration, not NNLO QCD",
                                                display_note="Controlled Born-decay illustration · not a QCD calculation",
                                                local_statistics={name: bin_diagnostics(
                                                    (events["y"] + np.arctanh(events["c"]))[h.sum(axis=1) > 0],
                                                    EDGES, fit.weights[h.sum(axis=1) > 0],
                                                    reference=reference.mean[1:, 0]) for name, fit in fits.items()},
                                                script_sha256=file_hash(__file__),
                                                inputs={name: file_hash(directory / name) for name in
                                                        ("prior.npz", "training.npz", "validation.npz", "selection.json")}))
    for name in MODEL_BLOCKS:
        delta = np.asarray(report["models"][name]["difference"]["mean"])[0]
        print(f"{name:10s}: held-out absolute-bin RMS residual {np.sqrt(np.mean(delta**2)):.4g} pb")
    print(f"Open {directory / 'report' / 'report.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage", choices=("all", "train", "validate"), default="all")
    args = parser.parse_args()
    if args.stage in {"all", "train"}:
        train(args.output)
    if args.stage in {"all", "validate"}:
        validate(args.output)


if __name__ == "__main__":
    main()
