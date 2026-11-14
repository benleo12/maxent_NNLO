"""Physics-facing diagnostics must distinguish a fitted marginal from a correlation."""
import numpy as np
import pytest

from maxent_upgrade import (bin_diagnostics, compute_fo_moments, joint_moment_residuals,
                            upgrade, weight_diagnostics)


def test_local_concentration_and_empty_reference_bins():
    values = np.array([0.2, 0.4, 1.0, 3.0])
    weights = np.array([1., 1., 8., 2.])
    edges = [0., 1., 2., 3., 4., 5.]
    result = bin_diagnostics(values, edges, weights, reference=np.ones(5))
    np.testing.assert_array_equal(result["counts"], [2, 1, 0, 1, 0])
    np.testing.assert_allclose(result["effective_events"], [2, 1, 0, 1, 0])
    np.testing.assert_allclose(result["largest_variance_fraction"], [.5, 1, 0, 1, 0])
    np.testing.assert_array_equal(result["empty_with_reference"], [False, False, True, False, True])
    scaled = bin_diagnostics(values, edges, weights * 123)
    for key in scaled:
        np.testing.assert_allclose(scaled[key], result[key])


def test_last_edge_included_and_out_of_range_events_not_misassigned():
    result = bin_diagnostics([-1., 0., 1., 2.], [0., .5, 1.], [1., 1., 1., 1.])
    np.testing.assert_array_equal(result["counts"], [1, 1])
    np.testing.assert_allclose(result["weight_fraction"], [.25, .25])


def test_retained_statistics_uses_weighted_prior():
    result = weight_diagnostics([1, 2, 3], [1, 2, 3])
    assert result["effective_fraction_retained"] == 1
    assert result["largest_variance_fraction"] == pytest.approx(9 / 14)
    assert result["effective_events"] == pytest.approx(36 / 14)
    with pytest.raises(ValueError):
        weight_diagnostics([1, 2], [1, 2, 3])


def test_unused_joint_moments_reveal_correlation_missing_from_separate_fit():
    x, y = np.meshgrid(np.linspace(-1, 1, 25), np.linspace(-1, 1, 25))
    events = dict(x=x.ravel(), y=y.ravel(), weight=np.ones(x.size))
    fo = dict(events, weight=np.exp(.2 * events["x"] + .7 * events["x"] * events["y"]))
    config = dict(born={k: dict(range=(-1, 1), map="lin") for k in ("x", "y")},
                  mixed={"xy": dict(observables=("x", "y"), n=2,
                                    range={k: (-1, 1) for k in ("x", "y")},
                                    map={k: "lin" for k in ("x", "y")})})
    moments = compute_fo_moments(fo, config, n_max=4)
    separate = upgrade(events, moments, dict(config, mixed={}))
    joint = upgrade(events, moments, config)
    before = joint_moment_residuals(events, separate.weights, moments, config)["xy"]
    after = joint_moment_residuals(events, joint.weights, moments, config)["xy"]
    expected = np.average(events["x"] * events["y"], weights=fo["weight"])
    assert before["moments"][0]["target"] == pytest.approx(expected)
    assert abs(before["moments"][0]["residual"]) > .05
    assert abs(after["moments"][0]["residual"]) < 1e-4
    assert after["moments"][0]["sample_error_fixed_weights"] > 0


def test_missing_joint_window_is_reported_without_a_fabricated_residual():
    events = dict(x=np.array([0., 1.]), y=np.array([0., 1.]))
    config = {"mixed": {"xy": dict(observables=("x", "y"), n=1,
                                    range={"x": (0, 5), "y": (0, 5)},
                                    map={"x": "lin", "y": "lin"},
                                    profile={"x": {"a": 2, "b": 3, "c": 5}})}}
    moments = {"mixed": {"xy": {"rate": .3, "values": {(1, 1): .2}}}}
    result = joint_moment_residuals(events, [1, 1], moments, config)["xy"]
    assert not result["available"] and result["moments"] == []
    assert result["target_window_rate"] == .3
