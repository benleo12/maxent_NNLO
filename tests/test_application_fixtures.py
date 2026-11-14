"""Bundled application inputs preserve sampling/normalization and really fit."""
import json
from pathlib import Path

import numpy as np
import pytest

from maxent_upgrade import load_moments, upgrade_with_uncertainties


DATA = Path(__file__).resolve().parents[1] / "examples/data/applications"


@pytest.mark.parametrize("process", ("ggh", "aa", "zj"))
def test_bundled_application_input_contract(process):
    with np.load(DATA / f"{process}_prior.npz", allow_pickle=False) as archive:
        events = dict(archive)
    metadata = json.loads(str(events["metadata"]))
    _, config = load_moments(process, check_seam=False)
    assert set(config["born"]) | set(config["recoil"]) <= set(events)
    assert len(events["event_id"]) == len(np.unique(events["event_id"])) == metadata["events"]
    np.testing.assert_allclose(events["weight"] * events["inclusion_probability"],
                               events["original_weight"] * events["source_normalization"], rtol=2e-15)
    assert np.isfinite(events["weight"]).all() and (events["weight"] > 0).all()
    if process == "aa":
        for source_index, source in enumerate(metadata["sources"]):
            selected = events["source_file_index"] == source_index
            np.testing.assert_allclose(events["source_normalization"][selected],
                                       source["sigma_pb"] / source["n_generated"])


def test_higgs_lower_window_can_come_from_main_table():
    with np.load(DATA / "ggh_prior.npz", allow_pickle=False) as archive:
        events = dict(archive)
    ensemble = upgrade_with_uncertainties(events, "ggh", sources=("window",))
    assert ensemble.weights["window"].shape == (2, len(events["weight"]))
    assert (ensemble.weights["window"] > 0).all()
    np.testing.assert_allclose(ensemble.weights["window"].sum(axis=1), 1, atol=1e-13)
    windows = ensemble.provenance["variations"]["window"]
    assert [v["configuration"]["recoil"]["pT_H"]["profile"]["a"] for v in windows] == [37., 72.]


def test_observable_selection_keeps_absolute_event_normalization():
    from examples.application_demo import selected_histogram
    from maxent_upgrade import EnsembleResult, UpgradeResult
    weights = np.array([.2, .3, .5])
    fit = UpgradeResult(weights, 1., 0., float("nan"), {}, {})
    ensemble = EnsembleResult(fit, 10., {}, {}, {}, {})
    histogram, _ = selected_histogram(ensemble, [30, 40, 50], [20, 40, 60],
                                      np.array([True, False, True]))
    np.testing.assert_allclose(histogram["central"], [.1, .25])
    assert histogram["selected_cross_section_pb"] == 7
    np.testing.assert_array_equal(fit.weights, [.2, .3, .5])
