"""Canonical lepton observables for the independent W example."""
import json
from pathlib import Path

import numpy as np
import pytest


def test_w_runner_keeps_signed_lepton_eta_alias():
    from examples.w_joint_refinement import prepare_events
    events = dict(weight=np.ones(2), m_w=np.full(2, 80.), y_abs=np.ones(2),
                  pt_w=np.ones(2), ctw1=np.ones(2), y_l=np.array([-1., 1.]))
    prepared, _ = prepare_events(events, "wp")
    np.testing.assert_array_equal(prepared["eta_l"], [-1., 1.])
    assert "eta_l" not in events
    events["l_born"] = np.array([[30., 0, -40., 50.], [30., 0, 40., 50.]])
    prepared, _ = prepare_events(events, "wp")
    np.testing.assert_allclose(prepared["eta_l"], [-np.log(3), np.log(3)])


@pytest.mark.parametrize("charge", ["wp", "wm"])
def test_independent_w_reference_uses_actual_channel_counts(charge):
    """Padded precision extensions must reproduce the published joint covariance."""
    directory = Path(__file__).resolve().parents[1] / "examples/data/w_joint"
    with np.load(directory / f"w_{charge}_validation_runs.npz") as runs, \
            np.load(directory / f"w_{charge}_training_runs.npz") as training, \
            np.load(directory / f"w_{charge}_validation.npz") as reference:
        metadata = json.loads(str(reference["metadata"]))
        np.testing.assert_array_equal(runs["labels"], metadata["labels"])
        np.testing.assert_array_equal(runs["channels"], training["channels"])
        means, covariances = [], []
        for index, channel in enumerate(runs["acc"]):
            selected = runs["seeds"][index] >= 0
            count = int(selected.sum())
            assert count == runs["n_runs"][index] and count > 1
            assert np.isfinite(channel[selected]).all()
            assert np.isnan(channel[~selected]).all()
            seeds = runs["seeds"][index, selected]
            assert len(np.unique(seeds)) == count
            assert not set(seeds) & set(training["seeds"][index])
            means.append(channel[selected].mean(axis=0))
            covariances.append(np.cov(channel[selected, :, 0], rowvar=False, ddof=1) / count)
        np.testing.assert_allclose(sum(means), reference["reference"], rtol=1e-11, atol=1e-10)
        np.testing.assert_allclose(sum(covariances), reference["reference_cov"], rtol=1e-11, atol=1e-10)
