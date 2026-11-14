"""Training-only orchestration for the prespecified W physical study.

The public builders and generic fit-response implementation do all numerical
construction. This file only supplies the declared candidate/selection rule,
full-prior selection, split-half input pooling and immutable output records.
No validation integration is opened here.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

ROOT = Path(os.environ.get("WPRED_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "package"))
sys.path.insert(0, str(ROOT / "package/examples"))
from maxent_upgrade import NumericalInput, fit_constraints, plan_observable
from w_physical_inputs import build_blocks, candidate_blocks, load_input
from w_predictions import observable_groups

CANDIDATES = ("anchor", "R", "D", "R+D", "M", "R+M", "N", "R+N",
              "G", "R+G", "W", "R+W", "J4", "R+J4", "H0", "H", "S0", "S")
ORDER = ("D", "M", "H", "S")
RULE = dict(order=list(ORDER), anchor="R", significance=.01, rank_cutoff=1e-6,
            l2=1e-4, criterion="resolved at every coherent scale using the central complete residual covariance",
            fit_closure="every imposed residual <= one central input standard error at every scale, or1e-10 absolute numerical tolerance; finite positive normalized weights",
            unsupported="retain an inconclusive classification; do not force an addition",
            stopping="alternating-seed half agreement;160,320,640 runs/channel at100k points; cap inconclusive")


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prior(path, charge):
    """Reapply the v17 baseline; the stored in_fid flag uses a different cut."""
    with np.load(path, allow_pickle=False) as z:
        mask = ((z["pt_l"] > 20) & (abs(z["y_l"]) < 2.5)
                & (z["pt_nu"] > 25) & (z["mt"] > 40)
                & (z["weight"] > 0) & np.isfinite(z["m_w"])
                & (z["charge"] == (1 if charge.lower() == "wp" else -1)))
        ids = np.flatnonzero(mask)
        events = {key: z[key][mask].astype(float) for key in
                  ("l_born", "nu", "weight", "m_w", "pt_w", "pt_l", "mt", "y_l")}
        events["y_abs"] = abs(z["y_w"][mask])
        meta = dict(source_file=str(path), source_sha256=sha(path), n_fid=len(ids),
                    selected_rows_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                    n_total=int(z["n_total"]),
                    original_fiducial_rate_pb=float(events["weight"].sum() / z["n_total"]))
    return events, ids, meta


def load_training(path, subset="full"):
    data = load_input(path)
    if data.role != "training":
        raise ValueError("training orchestration refuses validation inputs")
    if subset == "full":
        return data
    parity = {"even": 0, "odd": 1}[subset]
    with np.load(path, allow_pickle=False) as z:
        groups, seeds = [], []
        for a, s in zip(z["acc"], z["seeds"]):
            keep = np.flatnonzero(s >= 0)[parity::2]
            groups.append(a[keep])
            seeds.extend(s[keep].tolist())
    mean = sum(a.mean(axis=0) for a in groups)
    cov = sum(np.cov(a[:, :, 0], rowvar=False, ddof=1) / len(a) for a in groups)
    return NumericalInput(data.labels, mean, cov, data.sample_id + "-" + subset,
                          role="training", normalization=data.normalization,
                          scale_names=data.scale_names,
                          metadata=dict(data.metadata, subset=subset,
                                        subset_rule="alternating sorted physical seeds within each channel",
                                        selected_counts=[len(a) for a in groups]),
                          run_ids=tuple("nnlojet:%d" % seed for seed in seeds))


def diagnostic_record(diagnostic):
    pred = diagnostic["residual"]
    return dict(block=diagnostic["block"], labels=diagnostic["labels"],
                omitted_fitted_labels=diagnostic["omitted_fitted_labels"],
                mean=pred.mean, covariance=pred.covariance, test=diagnostic["test"])


def fit_summary(fit):
    report = fit.central.report
    errors = np.sqrt(np.maximum(np.diag(report["target_covariance"]), 0))
    residual = report["constraint_residuals"]
    pulls = np.divide(abs(residual), errors, out=np.full_like(residual, np.nan), where=errors > 0)
    return dict(n_events=len(fit.p), n_constraints=len(fit.labels), labels=fit.labels,
                rate_pb=fit.data.mean[fit.data.normalization],
                ess_retained=fit.central.effN_retained,
                ess_fraction=float(1 / (len(fit.p) * (fit.weights @ fit.weights))),
                q_over_p_range=report["q_over_p_range"],
                maximum_constraint_pull_per_scale=np.nanmax(pulls, axis=1),
                maximum_absolute_constraint_residual_per_scale=np.max(abs(residual), axis=1),
                constraint_residuals=residual, target_covariance=report["target_covariance"],
                zero_error_labels=report["zero_error_labels"],
                zero_error_residuals=report["zero_error_residuals"])


def fit_checks(fit):
    """Numerical target closure, not a certificate of predictive accuracy."""
    report = fit.central.report
    error = np.sqrt(np.maximum(np.diag(report["target_covariance"]), 0))
    tolerance = np.maximum(error, 1e-10)
    residual = abs(report["constraint_residuals"])
    positive = bool(np.isfinite(fit.scale_weights).all() and (fit.scale_weights > 0).all())
    normalized = bool(np.allclose(fit.scale_weights.sum(axis=1), 1, rtol=0, atol=1e-12))
    closure = np.all(residual <= tolerance, axis=1)
    return dict(passed=positive and normalized and bool(closure.all()),
                finite_positive_weights=positive, normalized_weights=normalized,
                target_closure_per_scale=closure,
                maximum_residual_over_tolerance_per_scale=np.max(residual / tolerance, axis=1),
                interpretation="numerical imposed-target review; no independent predictive-accuracy claim")


def prediction_columns(events):
    """Combine the public observable groups in the common reference order."""
    groups = observable_groups(events, exploratory=True)
    values = np.column_stack([g["values"] for g in groups.values()])
    labels = tuple(label for g in groups.values() for label in g["labels"])
    definition = dict(groups={key: g["definition"] for key, g in groups.items()},
                      role="prespecified validation plus separately identified old-cut exploratory ledger",
                      denominator="same original baseline")
    return values, labels, definition


def run_selection(events, data, charge, *, event_ids, prior_id, on_fit=None):
    """Portable fixed D→M→H→S rule, inspecting training inputs only.

    ``on_fit(name, fit, history)`` may save a fit. The caller owns any subsequent
    split-half comparison; this function never adapts its statistical thresholds.
    """
    blocks = build_blocks(events, data, charge)
    selected, history = "R", []
    fit = fit_constraints(events["weight"], data, candidate_blocks(blocks, charge, selected),
                          prior_id=prior_id, event_ids=event_ids, l2=RULE["l2"])
    if on_fit:
        on_fit(selected, fit, history)
    if not fit_checks(fit)["passed"]:
        return dict(rule=RULE, selected=selected, history=history,
                    status="anchor_target_closure_failed", checks=fit_checks(fit)), fit
    for block in ORDER:
        diagnostic = diagnostic_record(fit.diagnose(blocks[block], rank_cutoff=RULE["rank_cutoff"],
                                                    significance=RULE["significance"]))
        resolved = diagnostic["test"]["resolved"]
        step = dict(before=selected, block=block, diagnostic=diagnostic,
                    covariance_review_required=diagnostic["test"]["status"] == "discarded_modes_require_review",
                    decision="add" if resolved is True else ("retain" if resolved is False else "inconclusive"))
        history.append(step)
        if resolved is True:
            proposed = selected + "+" + block
            try:
                newer = fit_constraints(events["weight"], data, candidate_blocks(blocks, charge, proposed),
                                        prior_id=prior_id, event_ids=event_ids, l2=RULE["l2"])
            except Exception as exc:
                step.update(decision="fit_failed", failure=repr(exc))
            else:
                step["proposed_fit_checks"] = fit_checks(newer)
                if not step["proposed_fit_checks"]["passed"]:
                    step.update(decision="target_closure_failed", proposed=proposed)
                    if on_fit:
                        on_fit(proposed, newer, history)
                else:
                    selected, fit = proposed, newer
                    if on_fit:
                        on_fit(selected, fit, history)
        print(json.dumps(plain(dict(stage="selection", charge=charge, block=block,
                                    decision=step["decision"], selected=selected))), flush=True)
    return dict(rule=RULE, selected=selected, history=history,
                status="inconclusive" if any(s["decision"] in {"inconclusive", "fit_failed", "target_closure_failed"}
                                             or s["covariance_review_required"] for s in history)
                else "training_sequence_complete"), fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--charge", choices=("wp", "wm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset", choices=("full", "even", "odd"), default="full")
    parser.add_argument("--candidate", choices=CANDIDATES)
    args = parser.parse_args()
    started = time.monotonic()
    events, ids, prior_meta = load_prior(args.prior, args.charge)
    data = load_training(args.training, args.subset)
    out = args.output / args.charge.upper() / args.subset
    out.mkdir(parents=True, exist_ok=True)
    prior_id = "w-predictivity-full-prior-" + args.charge
    values, labels, definition = prediction_columns(events)
    plan = {"all": plan_observable(values, labels=labels, definition=definition)}
    del values
    def save(name, fit, history):
        stem = ("candidate_" if args.candidate else "sequence_") + name
        record = dict(name=name, charge=args.charge, subset=args.subset,
                      role="prespecified singleton control" if args.candidate else "training sequence candidate",
                      rule=RULE, prior=prior_meta, training_input_sha256=sha(args.training),
                      history=history, fit=fit_summary(fit), checks=fit_checks(fit), elapsed_s=time.monotonic() - started,
                      validation_status="unopened; this candidate precedes the final split-half stability decision")
        fit.freeze(out / (stem + ".npz"), selection=plain(record), validation_plan=plan)
        write_json(out / (stem + ".json"), record)
    if args.candidate:
        blocks = build_blocks(events, data, args.charge)
        fit = fit_constraints(events["weight"], data, candidate_blocks(blocks, args.charge, args.candidate),
                              prior_id=prior_id, event_ids=ids, l2=RULE["l2"])
        save(args.candidate, fit, [])
        print(json.dumps(plain(dict(candidate=args.candidate, **fit_summary(fit),
                                    elapsed_s=time.monotonic() - started))), flush=True)
    else:
        record, fit = run_selection(events, data, args.charge, event_ids=ids, prior_id=prior_id, on_fit=save)
        record.update(prior=prior_meta, subset=args.subset, training_input_sha256=sha(args.training),
                      elapsed_s=time.monotonic() - started)
        write_json(out / "SELECTION.json", record)
        print(json.dumps(plain(dict(charge=args.charge, subset=args.subset, selected=record["selected"],
                                    status=record["status"], elapsed_s=record["elapsed_s"]))), flush=True)


if __name__ == "__main__":
    main()
