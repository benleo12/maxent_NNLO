"""Common-target covariance, actual refit response and frozen validation contracts."""
import numpy as np
import pytest

from maxent_upgrade.numerical import NumericalInput, bootstrap_runs
from maxent_upgrade.study import (ConstraintBlock, fit_constraints, load_frozen,
                                 plan_observable, residual_test, stack_predictions, validate_prediction)


def problem(raw=None, prior=None):
    rng = np.random.default_rng(31)
    x, y = rng.uniform(-1, 1, size=(2, 360))
    p = np.exp(.2 * y) if prior is None else np.asarray(prior)
    p = p / p.sum()
    w = (x + 1) / 2
    g = (1 - w) * x
    part = p @ g / (1 - p @ w)
    raw = np.array([10., .8, 2.]) if raw is None else raw
    cov = np.array([[.02, .001, .003], [.001, .003, -.001], [.003, -.001, .004]])
    data = NumericalInput(("rate", "wx", "window"), raw, cov, "training-31")
    pd = (1 - raw[2] / raw[0]) / (1 - p @ w) * (g - part * (1 - w))
    block = ConstraintBlock("blend", x[:, None], np.array([[0, 1, -part]]), ("x",),
                            np.array([part]), pd[:, None], dict(definition="declared smooth continuation"))
    fit = fit_constraints(p, data, [block], prior_id="prior-31", l2=.03)
    h = np.column_stack([x * x, y + x**3])
    return fit, h, x, y


def test_raw_input_derivative_matches_refitting_and_normalization():
    fit, h, _, _ = problem()
    prediction = fit.predict(h, labels=("x2", "y_plus_x3"), definition="test")
    analytic = prediction.sources[fit.data.fingerprint]["jacobian"]
    finite = np.empty_like(analytic)
    raw = fit.data.mean[:, 0]
    for i in range(len(raw)):
        step = np.eye(len(raw))[i] * 2e-4
        upper = problem(raw + step)[0].predict(h, labels=prediction.labels, definition="test")
        lower = problem(raw - step)[0].predict(h, labels=prediction.labels, definition="test")
        finite[:, i] = (upper.mean[0] - lower.mean[0]) / 4e-4
    np.testing.assert_allclose(analytic, finite, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(prediction.covariance_numerical,
                               finite @ fit.data.covariance @ finite.T, rtol=2e-5, atol=2e-8)


def test_prior_response_includes_refitting_blend_and_empirical_ridge():
    fit, h, x, y = problem()
    prediction = fit.predict(h, labels=("x2", "y_plus_x3"), definition="test")
    influence = prediction.priors[fit.prior_key]["influence"]
    direction = x**2 + .7 * y
    step = 2e-4
    upper = problem(prior=fit.p * np.exp(step * direction))[0]
    lower = problem(prior=fit.p * np.exp(-step * direction))[0]
    finite = fit.data.mean[0, 0] * ((upper.weights - lower.weights) @ h) / (2 * step)
    np.testing.assert_allclose(influence.T @ direction, finite, rtol=2e-5, atol=3e-7)
    np.testing.assert_allclose(influence.sum(axis=0), 0, atol=1e-12)


def test_diagnostic_retains_fit_target_cross_covariance():
    fit, _, x, _ = problem()
    block = ConstraintBlock("unused", (x**2)[:, None], [[0, 1, .2]], ("x2",))
    diagnostic = fit.diagnose(block)
    pred = diagnostic["residual"]
    _, response, _ = fit._response(block.features)
    jac = (response @ fit.matrix - block.matrix) @ fit.data.normalization_jacobian
    np.testing.assert_allclose(pred.covariance_numerical, jac @ fit.data.covariance @ jac.T)
    naive = ((response @ fit.matrix) @ fit.data.normalized_covariance @
             (response @ fit.matrix).T + block.matrix @ fit.data.normalized_covariance @ block.matrix.T)
    assert not np.allclose(pred.covariance_numerical, naive, rtol=.01, atol=1e-12)


def test_paired_model_differences_and_transform_keep_common_fluctuations():
    fit, h, _, _ = problem()
    prediction = fit.predict(h, labels=("x2", "y_plus_x3"), definition="test")
    identical = prediction.difference(prediction)
    np.testing.assert_array_equal(identical.mean, 0)
    np.testing.assert_array_equal(identical.covariance, 0)
    prior = fit_constraints(fit.p, fit.data, [], prior_id="prior-31")
    baseline = prior.predict(h, labels=prediction.labels, definition="test")
    difference = prediction.difference(baseline)
    jac = (next(iter(prediction.sources.values()))["jacobian"]
           - next(iter(baseline.sources.values()))["jacobian"])
    influence = (next(iter(prediction.priors.values()))["influence"]
                 - next(iter(baseline.priors.values()))["influence"])
    expected = jac @ fit.data.covariance @ jac.T + influence.T @ influence
    np.testing.assert_allclose(difference.covariance, expected)
    a, b = prediction.mean[0]
    ratio = prediction.transform(lambda v: (v[:, 0] / v[:, 1])[:, None],
                                 [[1 / b, -a / b**2]], labels=("ratio",), definition="x2 / y_plus_x3")
    derivative = np.array([[1 / b, -a / b**2]])
    np.testing.assert_allclose(ratio.covariance, derivative @ prediction.covariance @ derivative.T)


def test_rank_deficient_residual_cannot_silently_pass():
    zero = residual_test([[1., 0]], np.zeros((2, 2)))
    assert zero["resolved"] is None and zero["status"] == "discarded_modes_require_review"
    singular = residual_test([[1., -1.]], [[1., 1.], [1., 1.]])
    assert singular["rank"] == 1 and singular["resolved"] is None
    assert singular["discarded_standardized_norm"][0] > 1
    empty = residual_test(np.empty((1, 0)), np.empty((0, 0)))
    assert empty["status"] == "no_unfitted_constraints" and empty["resolved"] is None
    with pytest.raises(ValueError, match="positive semidefinite"):
        residual_test([[0, 0]], [[1, 2], [2, 1]])


def test_scale_choices_are_separate_residuals_not_extra_statistical_samples():
    test = residual_test([[4], [0]], [[1]])
    assert test["rank"] == 1
    assert not test["resolved"]
    assert test["p_values"][0] < .01 and test["p_values"][1] == 1


def test_freeze_reload_and_independent_validation(tmp_path, monkeypatch):
    fit, h, _, _ = problem()
    labels, definition = ("x2", "y_plus_x3"), dict(observables=["x*x", "y+x**3"])
    plan = {"held_out": plan_observable(h, labels=labels, definition=definition)}
    archive = tmp_path / "frozen.npz"
    fit.freeze(archive, selection=dict(candidate="blend", rank_cutoff=1e-6), validation_plan=plan)
    with pytest.raises(FileExistsError):
        fit.freeze(archive, selection=dict(candidate="blend"), validation_plan=plan)
    monkeypatch.setattr("maxent_upgrade.study._newton_maxent", lambda *a, **k: pytest.fail("refitted"))
    loaded = load_frozen(archive, fit.p, fit.data, fit.blocks, prior_id=fit.prior_id)
    np.testing.assert_array_equal(loaded.weights, fit.weights)
    assert loaded.central.effN == fit.central.effN
    prediction = loaded.predict(h, labels=labels, definition=definition, validation_key="held_out")
    reference = NumericalInput(("rate",) + labels, [10, 3, .1], np.eye(3) * .01,
                               "independent-reference", role="validation")
    result = validate_prediction(prediction, reference, independent_of=[fit.data.sample_id],
                                 definition=definition)
    np.testing.assert_allclose(result["difference"].covariance,
                               prediction.covariance + np.eye(2) * .01)
    with pytest.raises(ValueError, match="frozen plan"):
        loaded.predict(h * 2, labels=labels, definition=definition, validation_key="held_out")
    with pytest.raises(ValueError, match="distinct reference"):
        validate_prediction(prediction, reference, independent_of=[], definition=definition)
    loaded.blocks[0].features[0, 0] += .01
    with pytest.raises(ValueError, match="has changed"):
        loaded.predict(h, labels=labels, definition=definition)


def test_validation_rejects_unfrozen_and_overlapping_runs(tmp_path):
    fit, h, _, _ = problem()
    labels, definition = ("x2", "y_plus_x3"), "held-out test"
    reference = NumericalInput(("rate",) + labels, [10, 3, .1], np.eye(3), "ref", role="validation")
    pred = fit.predict(h, labels=labels, definition=definition)
    with pytest.raises(ValueError, match="frozen"):
        validate_prediction(pred, reference, independent_of=[fit.data.sample_id], definition=definition)
    fit.data.run_ids = reference.run_ids = ("rng:7",)
    fit = fit_constraints(fit.p, fit.data, fit.blocks, prior_id=fit.prior_id, l2=fit.l2)
    fit.freeze(tmp_path / "fit.npz", selection={"model": "blend"},
               validation_plan={"test": plan_observable(h, labels=labels, definition=definition)})
    pred = fit.predict(h, labels=labels, definition=definition, validation_key="test")
    with pytest.raises(ValueError, match="share run"):
        validate_prediction(pred, reference, independent_of=[fit.data.sample_id], definition=definition)


def runs_fixture():
    rng = np.random.default_rng(14)
    acc = np.full((2, 7, 3, 2), np.nan)
    seeds = np.full((2, 7), -1, dtype=int)
    for c, n in enumerate((4, 7)):
        x = rng.normal(size=n)
        acc[c, :n, :, 0] = np.column_stack([3 + x, 2 * x, -x])
        acc[c, :n, :, 1] = acc[c, :n, :, 0] * 1.1
        seeds[c, :n] = np.arange(n) + c * 100
    return acc, seeds


def test_unequal_run_counts_full_covariance_and_whole_vector_bootstrap():
    acc, seeds = runs_fixture()
    data = NumericalInput.from_runs(("rate", "a", "b"), acc, seeds=seeds,
                                    channels=("one", "two"), sample_id="runs")
    expected = np.cov(acc[0, :4, :, 0], rowvar=False) / 4 + np.cov(acc[1, :, :, 0], rowvar=False) / 7
    np.testing.assert_allclose(data.covariance, expected)
    replicas = bootstrap_runs(acc, seeds=seeds, channels=("one", "two"), replicas=100)
    np.testing.assert_allclose(replicas[:, 1], -2 * replicas[:, 2], atol=1e-14)
    np.testing.assert_allclose(replicas[:, 0], 6 + replicas[:, 1] / 2, atol=1e-14)
    assert data.metadata["channel_counts"] == {"one": 4, "two": 7}


@pytest.mark.parametrize("bad", ["fractional", "negative", "overlap", "partial"])
def test_invalid_run_identity_or_padding_is_rejected(bad):
    acc, seeds = runs_fixture()
    if bad == "fractional":
        seeds = seeds.astype(float) + .2
    elif bad == "negative":
        seeds[0, 4] = -2
    elif bad == "overlap":
        seeds[1, 0] = 0
    else:
        acc[0, 4, 0, 0] = 0
    with pytest.raises(ValueError):
        NumericalInput.from_runs(("rate", "a", "b"), acc, seeds=seeds,
                                 channels=("one", "two"), sample_id="runs")


def test_pairing_rejects_reordered_or_subset_priors_and_overlapping_runs():
    fit, h, _, _ = problem()
    first = fit.predict(h, labels=("a", "b"), definition="test")
    for ids in (fit.event_ids[::-1], fit.event_ids[:-1]):
        second_fit = fit_constraints(fit.p[:len(ids)], fit.data, [], prior_id=fit.prior_id, event_ids=ids)
        second = second_fit.predict(h[:len(ids)], labels=("a", "b"), definition="test")
        with pytest.raises(ValueError, match="align IDs"):
            stack_predictions([first, second])
    key = next(iter(first.sources))
    first.sources[key]["identity"]["run_ids"] = ["rng:3"]
    other_data = NumericalInput(fit.data.labels, fit.data.mean, fit.data.covariance,
                                "other-name", run_ids=("rng:3",))
    second = fit_constraints(fit.p, other_data, [], prior_id=fit.prior_id).predict(
        h, labels=("a", "b"), definition="test")
    with pytest.raises(ValueError, match="overlap in run"):
        stack_predictions([first, second])


def test_reused_diagnostic_label_must_match_its_fitted_definition():
    fit, _, x, _ = problem()
    changed = ConstraintBlock("changed", x[:, None] * 2, fit.matrix, ("x",), fit.offset)
    with pytest.raises(ValueError, match="different feature"):
        fit.diagnose(changed)


def test_exact_targets_do_not_report_zero_pull_for_nonzero_ridge_residual():
    x = np.linspace(-1, 1, 100)
    data = NumericalInput(("rate", "x"), [1, .3], np.zeros((2, 2)), "exact")
    block = ConstraintBlock.from_input("x", x[:, None], data, labels=("x",))
    fit = fit_constraints(np.ones(len(x)), data, [block], prior_id="events", l2=.1)
    assert np.isnan(fit.central.max_pull)
    assert abs(fit.central.report["zero_error_residuals"][0, 0]) > .001
    assert "zero MC error" in fit.central.summary()
    zero = NumericalInput(("rate", "x"), [1, 0], np.zeros((2, 2)), "exact-zero")
    zero_block = ConstraintBlock.from_input("x", x[:, None], zero, labels=("x",))
    zero_fit = fit_constraints(np.exp(x), zero, [zero_block], prior_id="events", l2=.1)
    assert np.isnan(zero_fit.central.closure) and "zero MC error" in zero_fit.central.summary()


def test_zero_ridge_dependent_response_and_overflowed_prior_fail_clearly():
    x = np.linspace(-1, 1, 100)
    data = NumericalInput(("rate", "x", "x_again"), [1, .1, .1], np.eye(3) * .01, "data")
    block = ConstraintBlock.from_input("duplicate", np.column_stack([x, x]), data,
                                       labels=("x", "x_again"))
    fit = fit_constraints(np.ones(len(x)), data, [block], prior_id="events", l2=0)
    with pytest.raises(ValueError, match="not identifiable"):
        fit.predict(x**2, labels=("x2",), definition="test")
    with pytest.raises(ValueError, match="sum of prior"):
        fit_constraints([1e308, 1e308], data, [], prior_id="events")
