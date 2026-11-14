"""Training-only W selection shared with the independent physics study.

This is a process-specific prescription built from the public study interface.
It keeps the declared block order, rank/significance thresholds and numerical
closure checks. References and validation cuts never enter the choice.
"""

import json
import numpy as np
from maxent_upgrade import NumericalInput, fit_constraints

if __package__:
    from .w_physical_inputs import build_blocks, candidate_blocks, load_input
else:
    from w_physical_inputs import build_blocks, candidate_blocks, load_input


CANDIDATES = (
    "anchor",
    "R",
    "D",
    "R+D",
    "M",
    "R+M",
    "N",
    "R+N",
    "G",
    "R+G",
    "W",
    "R+W",
    "J4",
    "R+J4",
    "H0",
    "H",
    "S0",
    "S",
)


ORDER = ("D", "M", "H", "S")


RULE = dict(
    order=list(ORDER),
    anchor="R",
    significance=0.01,
    rank_cutoff=1e-6,
    l2=1e-4,
    criterion="resolved at every coherent scale using the central complete residual covariance",
    fit_closure="every imposed residual <= one central input standard error at every scale, or1e-10 absolute numerical tolerance; finite positive normalized weights",
    unsupported="retain an inconclusive classification; do not force an addition",
    stopping="alternating-seed half agreement;160,320,640 runs/channel at100k points; cap inconclusive",
)


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
            present = np.flatnonzero(s >= 0)
            keep = present[np.argsort(s[present])][parity::2]
            if len(keep) < 2:
                raise ValueError("each training half needs at least two runs per additive channel")
            groups.append(a[keep])
            seeds.extend(s[keep].tolist())
    mean = sum(a.mean(axis=0) for a in groups)
    cov = sum(np.cov(a[:, :, 0], rowvar=False, ddof=1) / len(a) for a in groups)
    return NumericalInput(
        data.labels,
        mean,
        cov,
        data.sample_id + "-" + subset,
        role="training",
        normalization=data.normalization,
        scale_names=data.scale_names,
        metadata=dict(
            data.metadata,
            subset=subset,
            subset_rule="alternating sorted physical seeds within each channel",
            selected_counts=[len(a) for a in groups],
        ),
        run_ids=tuple("nnlojet:%d" % seed for seed in seeds),
    )


def diagnostic_record(diagnostic):
    pred = diagnostic["residual"]
    return dict(
        block=diagnostic["block"],
        labels=diagnostic["labels"],
        omitted_fitted_labels=diagnostic["omitted_fitted_labels"],
        mean=pred.mean,
        covariance=pred.covariance,
        test=diagnostic["test"],
    )


def fit_checks(fit):
    """Numerical target closure, not a certificate of predictive accuracy."""
    report = fit.central.report
    error = np.sqrt(np.maximum(np.diag(report["target_covariance"]), 0))
    tolerance = np.maximum(error, 1e-10)
    residual = abs(report["constraint_residuals"])
    positive = bool(np.isfinite(fit.scale_weights).all() and (fit.scale_weights > 0).all())
    normalized = bool(np.allclose(fit.scale_weights.sum(axis=1), 1, rtol=0, atol=1e-12))
    closure = np.all(residual <= tolerance, axis=1)
    return dict(
        passed=positive and normalized and bool(closure.all()),
        finite_positive_weights=positive,
        normalized_weights=normalized,
        target_closure_per_scale=closure,
        maximum_residual_over_tolerance_per_scale=np.max(residual / tolerance, axis=1),
        interpretation="numerical imposed-target review; no independent predictive-accuracy claim",
    )


def run_selection(events, data, charge, *, event_ids, prior_id, on_fit=None):
    """Portable fixed D→M→H→S rule, inspecting training inputs only.

    ``on_fit(name, fit, history)`` may save a fit. The caller owns any subsequent
    split-half comparison; this function never adapts its statistical thresholds.
    """
    blocks = build_blocks(events, data, charge)
    selected, history = "R", []
    fit = fit_constraints(
        events["weight"],
        data,
        candidate_blocks(blocks, charge, selected),
        prior_id=prior_id,
        event_ids=event_ids,
        l2=RULE["l2"],
    )
    if on_fit:
        on_fit(selected, fit, history)
    if not fit_checks(fit)["passed"]:
        return dict(
            rule=RULE,
            selected=selected,
            history=history,
            status="anchor_target_closure_failed",
            checks=fit_checks(fit),
        ), fit
    for block in ORDER:
        diagnostic = diagnostic_record(
            fit.diagnose(
                blocks[block], rank_cutoff=RULE["rank_cutoff"], significance=RULE["significance"]
            )
        )
        resolved = diagnostic["test"]["resolved"]
        step = dict(
            before=selected,
            block=block,
            diagnostic=diagnostic,
            covariance_review_required=diagnostic["test"]["status"]
            == "discarded_modes_require_review",
            decision="add"
            if resolved is True
            else ("retain" if resolved is False else "inconclusive"),
        )
        history.append(step)
        if resolved is True:
            proposed = selected + "+" + block
            try:
                newer = fit_constraints(
                    events["weight"],
                    data,
                    candidate_blocks(blocks, charge, proposed),
                    prior_id=prior_id,
                    event_ids=event_ids,
                    l2=RULE["l2"],
                )
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
        print(
            json.dumps(
                plain(
                    dict(
                        stage="selection",
                        charge=charge,
                        block=block,
                        decision=step["decision"],
                        selected=selected,
                    )
                )
            ),
            flush=True,
        )
    return dict(
        rule=RULE,
        selected=selected,
        history=history,
        status="inconclusive"
        if any(
            s["decision"] in {"inconclusive", "fit_failed", "target_closure_failed"}
            or s["covariance_review_required"]
            for s in history
        )
        else "training_sequence_complete",
    ), fit
