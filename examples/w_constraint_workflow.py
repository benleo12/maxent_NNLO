"""A real W sample: training diagnostics, frozen choices and new predictions.

    python examples/w_constraint_workflow.py wm --output w_demo

The compact demonstration has its own training choice and event statistics.
It does not reproduce the full-study choice automatically. Numerical uncertainty
uses the complete common-integral and refitted-prior response; the seven scale
choices remain coherent variations. The independent reference is opened only
after every candidate and the declared observable columns have been frozen.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import time

if __name__ == "__main__":
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "2")

import numpy as np

from maxent_upgrade import (
    bin_diagnostics,
    fit_constraints,
    load_frozen,
    plan_observable,
    validate_prediction,
)
from maxent_upgrade.provenance import canonical_json, file_hash
from maxent_upgrade.study_report import write_study_report

if __package__:
    from .w_coordinates import coordinates
    from .w_physical_inputs import DATA, build_blocks, candidate_blocks, load_input
    from .w_predictions import ETA_EDGES, acceptances, observable_groups, prediction_columns
    from .w_selection import ORDER, RULE, fit_checks, load_training, plain, run_selection
else:
    from w_coordinates import coordinates
    from w_physical_inputs import DATA, build_blocks, candidate_blocks, load_input
    from w_predictions import ETA_EDGES, acceptances, observable_groups, prediction_columns
    from w_selection import ORDER, RULE, fit_checks, load_training, plain, run_selection


BLOCK_NAMES = {
    "D": "Higher angular moments",
    "M": "Mass–angle correlations",
    "H": "Hard-recoil spin",
    "S": "Recoil-suppressed spin",
}
SCALE_FACTORS = [[1, 1], [2, 2], [0.5, 0.5], [2, 1], [1, 2], [0.5, 1], [1, 0.5]]
CENTRAL_SCALE = "charged-lepton plus neutrino invariant mass"


def load_events(path, charge=None):
    with np.load(path, allow_pickle=False) as archive:
        events = dict(archive)
    if "event_id" not in events:
        raise ValueError("retain original unique event_id values in the prior file")
    if charge is not None and "charge" in events:
        if not np.all(events["charge"] == (1 if charge == "wp" else -1)):
            raise ValueError("the event sample must contain only the requested W charge")
    return events


def check_input_charge(data, charge):
    if str(data.metadata.get("charge", "")).lower() != charge:
        raise ValueError("the numerical input must have the requested W charge")


def train(charge, output, prior, training):
    started = time.perf_counter()
    events, data = load_events(prior, charge), load_training(training)
    check_input_charge(data, charge)
    output.mkdir(parents=True, exist_ok=False)
    prior_id = f"w-compact-{charge}-" + file_hash(prior)
    blocks = build_blocks(events, data, charge)
    fits = {
        "Prior": fit_constraints(
            events["weight"], data, [], prior_id=prior_id, event_ids=events["event_id"]
        ),
        "anchor": fit_constraints(
            events["weight"],
            data,
            candidate_blocks(blocks, charge, "anchor"),
            prior_id=prior_id,
            event_ids=events["event_id"],
            l2=RULE["l2"],
        ),
    }
    selection, _ = run_selection(
        events,
        data,
        charge,
        event_ids=events["event_id"],
        prior_id=prior_id,
        on_fit=lambda name, fit, history: fits.update({name: fit}),
    )
    halves = {}
    for subset in ("even", "odd"):
        half_record, half_fit = run_selection(
            events,
            load_training(training, subset),
            charge,
            event_ids=events["event_id"],
            prior_id=prior_id,
        )
        halves[subset] = half_record
        del half_fit
    stable = all(
        r["status"] == "training_sequence_complete" and r["selected"] == selection["selected"]
        for r in [selection, *halves.values()]
    )
    selection.update(
        charge=charge,
        anchor="R",
        split_halves=halves,
        stable=stable,
        choice_status="stable_training_choice" if stable else "inconclusive_training_choice",
        demonstration="compact prior; its choice need not match the full production",
        extension="no automatic new integration; retain an inconclusive result at the supplied budget",
        training_sha256=file_hash(training),
        prior_sha256=file_hash(prior),
        prior_id=prior_id,
        validation_status="unopened",
        candidates=list(fits),
        numerical_environment=dict(
            numpy=np.__version__,
            runtime=platform.python_version(),
            system=platform.platform(),
            thread_environment={
                k: os.environ.get(k)
                for k in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        ),
        candidate_checks={k: fit_checks(v) for k, v in fits.items() if k != "Prior"},
    )
    values, labels, definition = prediction_columns(events)
    plan = {"all": plan_observable(values, labels=labels, definition=definition)}
    weights = {
        key: events[key]
        for key in (
            "event_id",
            "source_index",
            "weight",
            "original_weight",
            "inclusion_probability",
            "inverse_inclusion_probability",
        )
        if key in events
    }
    (output / "frozen").mkdir()
    for i, (name, fit) in enumerate(fits.items()):
        fit.freeze(
            output / "frozen" / f"candidate_{i}.npz",
            selection=plain(dict(selection, candidate=name)),
            validation_plan=plan,
            portable=True,
        )
        weights[f"candidate_{i}_weights_pb"] = (
            fit.scale_weights * data.mean[data.normalization, :, None]
        )
    chosen = fits[selection["selected"]]
    weights.update(
        weight_upgraded=chosen.scale_weights[0] * data.mean[data.normalization, 0],
        weight_scale=chosen.scale_weights[1:] * data.mean[data.normalization, 1:, None],
        metadata=np.array(
            canonical_json(
                plain(
                    dict(
                        candidates=list(fits),
                        selected=selection["selected"],
                        scale_names=data.scale_names,
                        units="absolute pb per event; replace original weights",
                        prior_weights="weight retains the supplied prior weight; original_weight and inclusion columns, when present, retain the source generator and sampling information",
                        numerical_uncertainty="full target/prior response in the prediction reports, not event-wise envelope weights",
                    )
                )
            )
        ),
    )
    np.savez_compressed(output / "weights.npz", **weights)
    selection["elapsed_training_seconds"] = time.perf_counter() - started
    (output / "selection.json").write_text(canonical_json(plain(selection)) + "\n")
    print(f"Frozen compact {charge}: {selection['selected']} ({selection['choice_status']}).")


def projection(prediction, indices, labels, definition):
    matrix = np.eye(len(prediction.labels))[indices]
    return prediction.transform(
        lambda a: a[:, indices], matrix, labels=labels, definition=definition
    )


def load_candidate(output, *, prior=None, name=None):
    """Open a saved compact W candidate for another analysis, without rebuilding it.

    Returns the original event arrays and restored fit. By default this opens
    the selected candidate and the bundled prior; pass the same original prior
    path for a user-supplied sample. Its complete file identity must still match.
    """
    output = Path(output)
    selection = json.loads((output / "selection.json").read_text())
    prior = Path(prior) if prior else DATA / f"w_{selection['charge']}_prior.npz"
    if file_hash(prior) != selection["prior_sha256"]:
        raise ValueError("the original event file does not match this saved W study")
    events = load_events(prior, selection["charge"])
    index = selection["candidates"].index(name or selection["selected"])
    fit = load_frozen(
        output / "frozen" / f"candidate_{index}.npz",
        events["weight"],
        prior_id=selection["prior_id"],
        event_ids=events["event_id"],
    )
    return events, fit


def validate(charge, output, prior, training, reference_path):
    started = time.perf_counter()
    selection = json.loads((output / "selection.json").read_text())
    if selection["charge"] != charge:
        raise ValueError("the saved study has a different W charge")
    if selection["prior_sha256"] != file_hash(prior) or selection["training_sha256"] != file_hash(
        training
    ):
        raise ValueError("frozen prior/training files have changed")
    events, data = load_events(prior, charge), load_training(training)
    check_input_charge(data, charge)
    blocks, fits, diagnostics = build_blocks(events, data, charge), {}, {}
    for i, name in enumerate(selection["candidates"]):
        selected_blocks = [] if name == "Prior" else candidate_blocks(blocks, charge, name)
        frozen_path = output / "frozen" / f"candidate_{i}.npz"
        with np.load(frozen_path, allow_pickle=False) as archive:
            portable = "portable" in json.loads(str(archive["manifest"]))
        fits[name] = load_frozen(
            frozen_path,
            events["weight"],
            data,
            None if portable else selected_blocks,
            prior_id=selection["prior_id"],
            event_ids=events["event_id"],
        )
        diagnostics[name] = {
            BLOCK_NAMES[key]: fits[name].diagnose(
                blocks[key], rank_cutoff=RULE["rank_cutoff"], significance=RULE["significance"]
            )
            for key in ORDER
        }
    # Every candidate has been restored and checked before reference means enter.
    reference = load_input(reference_path)
    check_input_charge(reference, charge)
    if reference.role != "validation":
        raise ValueError("an independent validation input is required")
    if (
        reference.metadata.get("scale_factors_muR_muF") != SCALE_FACTORS
        or reference.metadata.get("central_scale") != CENTRAL_SCALE
    ):
        raise ValueError(
            "reference scale prescription/order must match the audited W training inputs"
        )
    values, labels, definition = prediction_columns(events)
    if tuple(reference.labels) != labels:
        raise ValueError("reference column ordering differs from the declared prediction vector")
    comparisons, cuts, arrays, rate_predictions = {}, {}, {}, {}
    groups = observable_groups(events)
    eta_group = groups["eta32"]
    eta_indices = [labels.index(label) for label in eta_group["labels"]]
    selected_raw = anchor_raw = None
    for i, (name, fit) in enumerate(fits.items()):
        prediction = fit.predict(values, labels=labels, definition=definition, validation_key="all")
        validation = validate_prediction(
            prediction, reference, independent_of=[data.sample_id], definition=definition
        )
        comparisons[name] = {
            kind: projection(
                validation[kind], eta_indices, eta_group["labels"], eta_group["definition"]
            )
            for kind in ("prediction", "reference", "difference")
        }
        comparisons[name]["definition"] = eta_group["definition"]
        rate_results = {
            kind: acceptances(
                projection(
                    validation[kind],
                    list(range(7)),
                    groups["cuts"]["labels"],
                    groups["cuts"]["definition"],
                )
            )
            for kind in ("prediction", "reference")
        }
        rate_results["difference"] = rate_results["prediction"].difference(
            rate_results["reference"]
        )
        cuts[name] = {kind: result.summary() for kind, result in rate_results.items()}
        rate_predictions[name] = rate_results["prediction"]
        for kind, result in validation.items():
            if kind in {"prediction", "reference", "difference"}:
                arrays[f"candidate_{i}_{kind}_mean"] = result.mean
                arrays[f"candidate_{i}_{kind}_cov_numerical"] = result.covariance_numerical
                arrays[f"candidate_{i}_{kind}_cov_prior"] = result.covariance_prior
        if name == selection["selected"]:
            selected_raw = prediction
        if name == "R":
            anchor_raw = prediction
    change = selected_raw.difference(anchor_raw)
    for name, result in rate_predictions.items():
        cuts[name]["paired_change_from_R"] = result.difference(rate_predictions["R"]).summary()
    arrays.update(
        selected_change_from_R=change.mean,
        selected_change_covariance=change.covariance,
        metadata=np.array(
            canonical_json(
                plain(
                    dict(
                        labels=labels,
                        definition=definition,
                        selection=selection,
                        reference_sha256=file_hash(reference_path),
                    )
                )
            )
        ),
    )
    np.savez_compressed(output / "predictions.npz", **arrays)
    (output / "acceptances.json").write_text(canonical_json(plain(cuts)) + "\n")
    accepted = eta_group["values"].sum(axis=1) > 0
    lepton_rapidity = coordinates(events["l_born"], events["nu"])["y_l"]
    reference_bins = reference.mean[eta_indices, 0]
    local_statistics = {
        name: bin_diagnostics(
            np.abs(lepton_rapidity)[accepted],
            ETA_EDGES,
            fit.weights[accepted],
            reference=reference_bins if np.all(reference_bins >= 0) else None,
        )
        for name, fit in fits.items()
    }
    display = {
        name: {
            "Prior": "Prior",
            "anchor": "Blended recoil",
            "R": "Windowed recoil",
            "R+H": "Hard-recoil spin",
            "R+S": "Recoil-suppressed spin",
            "R+H+S": "Recoil–spin moments",
        }.get(name, "Added blocks: " + name.removeprefix("R+").replace("+", ", "))
        for name in fits
    }
    if selection["selected"] != "R":
        display[selection["selected"]] = (
            "Recoil–spin moments"
            if selection["selected"] == "R+H+S"
            else "Additional decay constraints"
            if selection["stable"]
            else "Combined candidate"
        )
    write_study_report(
        output / "report",
        title=f"W{'+' if charge == 'wp' else '−'}: physical constraints and lepton predictions",
        models={display[k]: v for k, v in fits.items()},
        diagnostics={display[k]: v for k, v in diagnostics.items()},
        validations={display[k]: v for k, v in comparisons.items()},
        edges=ETA_EDGES,
        xlabel=r"Charged-lepton $|\eta_\ell|$ after $p_{T,\ell}>32$ GeV",
        ylabel=r"$d\sigma/d|\eta_\ell|$ [pb]",
        selected=display[selection["selected"]],
        anchor=display["R"],
        color_overrides={
            display["anchor"]: "#23658b",
            display["R"]: "#cf7a2b",
            display[selection["selected"]]: "#bd3c3c",
        },
        comparison=list(
            dict.fromkeys([display[k] for k in ("Prior", "anchor", "R", selection["selected"])])
        ),
        provenance=dict(
            display_note=f"Real W{'+' if charge == 'wp' else '−'} compact demonstration · "
            + selection["choice_status"].replace("_", " "),
            selection=selection,
            local_statistics=local_statistics,
            local_statistics_scope="Counts and concentration among events passing pT_l>32 GeV; these do not certify unobserved phase-space support.",
            lepton_convention="The displayed eta uses the massless-lepton convention; booked and shower values are both charged-lepton rapidity y_l.",
            reference_sha256=file_hash(reference_path),
            scale_factors_muR_muF=SCALE_FACTORS,
            central_scale=CENTRAL_SCALE,
            script_sha256=file_hash(__file__),
            elapsed_validation_seconds=time.perf_counter() - started,
        ),
    )
    print(f"Open {output / 'report' / 'report.html'}; absolute weights: {output / 'weights.npz'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("charge", choices=("wp", "wm"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "train", "validate"), default="all")
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    prior = args.prior or DATA / f"w_{args.charge}_prior.npz"
    training = args.training or DATA / "initial160" / f"T_{args.charge.upper()}.npz"
    reference = args.reference or DATA / "reference" / f"V_{args.charge.upper()}.npz"
    if args.stage in {"all", "train"}:
        train(args.charge, args.output, prior, training)
    if args.stage in {"all", "validate"}:
        validate(args.charge, args.output, prior, training, reference)


if __name__ == "__main__":
    main()
