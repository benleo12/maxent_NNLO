"""Coherent rate/shape propagation and the user-facing uncertainty workflow."""
import json

import numpy as np
import pytest

from maxent_upgrade import (
    Variation, compute_fo_moments, histogram_bands, released,
    upgrade_ensemble, upgrade_with_uncertainties,
)
from maxent_upgrade.cli import main


def problem(tilt=0.3):
    x = np.linspace(-1, 1, 256)
    events = dict(x=x, weight=np.ones(len(x)), event_id=np.arange(len(x)))
    config = {"born": {"x": {"map": "lin", "range": (-1, 1)}}}
    moments = compute_fo_moments(dict(events, weight=np.exp(tilt*x)), config, n_max=3)
    return events, moments, config


def test_custom_ensemble_preserves_rate_shape_and_fixed_orders():
    events, moments, config = problem()
    config["L2"] = 0.0  # check the exact exponential family without ridge bias
    _, shifted, _ = problem(0.5)
    variations = {
        "scale": [Variation(shifted, cross_section=12), Variation(cross_section=8)],
        "window": [Variation(moments, config=config)],
        "shower": [Variation(prior_weight=np.exp(-0.2*events["x"]))],
    }
    calls = []
    result = upgrade_ensemble(events, moments, config, variations, cross_section=10,
                              progress=lambda label, r: calls.append(r))
    for r in calls:
        assert r.chosen_moments == result.central.chosen_moments
    saved = result.arrays()
    np.testing.assert_allclose(saved["weight_scale"].sum(axis=1), [12, 8])
    np.testing.assert_allclose(result.arrays(unit_area=True)["weight_scale"].sum(axis=1), 1)
    edges = np.array([-1, 0, 1.])
    a = histogram_bands(events["x"], edges, saved)
    b = result.histogram(events["x"], edges)
    for key in a:
        np.testing.assert_allclose(a[key], b[key])
    assert a["central"].sum() == pytest.approx(10)
    assert (a["scale_low"] <= a["central"]).all()
    assert (a["scale_high"] >= a["central"]).all()
    np.testing.assert_allclose(a["window_low"], a["central"], atol=2e-7)
    # Changed positive prior is corrected to the SAME exponential-family target.
    np.testing.assert_allclose(a["shower_high"], a["central"], atol=3e-5)


def test_histogram_bands_use_coherent_histograms_and_full_normalization():
    values = np.array([0.2, 0.8, 1.2, 1.8])
    saved = dict(weight_upgraded=np.array([1., 1., 1., 1.]),
                 uncertainty_sources=np.array(["scale", "stat", "pdf"]),
                 uncertainty_methods=np.array(["envelope", "std", "hessian68"]),
                 weight_scale=np.array([[2, 0, 0, 2], [0, 2, 2, 0.]]),
                 weight_stat=np.array([[1, 1, 1, 1], [2, 2, 2, 2.]]),
                 weight_pdf=np.array([[2, 1, 1, 1], [0.5, 1, 1, 1.]]))
    hist = histogram_bands(values, [0, 1], saved, unit_area=False)
    # Taking per-event envelopes would yield [0,4], incorrectly. Coherent sums both equal 2.
    np.testing.assert_allclose(hist["scale_low"], [2])
    np.testing.assert_allclose(hist["scale_high"], [2])
    np.testing.assert_allclose(hist["stat_high"], [2 + np.sqrt(2)])
    np.testing.assert_allclose(hist["pdf_high"], [2 + 1/1.645])
    np.testing.assert_allclose(hist["pdf_low"], [2 - 0.5/1.645])
    hist = histogram_bands(values, [0, 1], saved, unit_area=True)
    np.testing.assert_allclose(hist["central"], [0.5])
    np.testing.assert_allclose(hist["stat_high"], [0.5])


def test_released_driver_and_cli_round_trip(tmp_path, monkeypatch):
    events, moments, config = problem()
    calls = []
    def loader(process, **kwargs):
        calls.append(kwargs)
        return moments, config
    monkeypatch.setattr(released, "load_moments", loader)
    monkeypatch.setattr(released, "table", lambda p: dict(
        fiducial="synthetic", scales=list(range(7)), n_independent_runs=5,
        default_constraints={"recoil": ["x"]}))
    monkeypatch.setattr(released, "sigma_fid", lambda p: {"value": 10})
    monkeypatch.setattr(released, "cross_section", lambda p, **kw: 10 + kw.get("scale", 0))
    monkeypatch.setattr(released, "windows", lambda p: {"down": {}, "up": {}})
    monkeypatch.setattr(released, "_window", lambda *a: ({"x": {}}, {}, {}, "custom"))
    monkeypatch.setattr(released, "pdf_table", lambda p: {"n_members": 3})
    result = upgrade_with_uncertainties(events, "custom", n_boot=3, seed=17)
    assert {k: len(v) for k, v in result.weights.items()} == dict(scale=6, stat=3, window=2, pdf=2)
    replicas = [c["runs"] for c in calls if "runs" in c]
    rng = np.random.default_rng(17)
    for runs in replicas:
        np.testing.assert_array_equal(runs, rng.integers(0, 5, 5))
    prior, posterior, output = (tmp_path / n for n in ("prior.npz", "posterior.npz", "hist.npz"))
    np.savez(prior, **events)
    main(["reweight", "custom", str(prior), str(posterior), "--uncertainties", "--bootstrap", "3", "--seed", "17"])
    with np.load(posterior) as saved:
        for k, v in events.items():
            np.testing.assert_array_equal(saved[k], v)
        for k, v in result.arrays().items():
            if k == "provenance_json":
                actual = json.loads(str(saved[k]))
                expected = json.loads(str(v))
                input_file = actual.pop("input_file")
                assert input_file["name"] == prior.name and len(input_file["sha256"]) == 64
                assert actual == expected
            else:
                np.testing.assert_array_equal(saved[k], v)
    main(["histogram", str(posterior), "x", str(output), "--bins", "2", "--range", "-1", "1"])
    with np.load(output) as hist:
        assert hist["central"].sum() == pytest.approx(10)
        assert "pdf_high" in hist and "window_low" in hist


def test_unsupported_or_invalid_ensembles_are_explicit():
    events, moments, config = problem()
    for options in ({"n_boot": 1}, {"window": "up"}, {"sources": ["made_up"]}):
        with pytest.raises(ValueError):
            upgrade_with_uncertainties(events, "dy", **options)
    for source, shifts in (("stat", [Variation()]), ("pdf", [Variation()]), ("scale", [])):
        with pytest.raises(ValueError):
            upgrade_ensemble(events, moments, config, {source: shifts})


def test_changed_measurement_functions_start_cold(monkeypatch):
    import maxent_upgrade.uncertainty as module
    events, moments, config = problem()
    original = module.upgrade
    calls = []

    def record(events, moments, config):
        calls.append(config)
        return original(events, moments, config)

    monkeypatch.setattr(module, "upgrade", record)
    upgrade_ensemble(events, moments, config,
                     {"scale": [Variation()], "window": [Variation(config=config)]})
    assert "lam0" in calls[1]
    assert "lam0" not in calls[2]
    assert calls[1]["fixed_orders"] == calls[2]["fixed_orders"]


def test_sample_error_and_unit_area_file_metadata():
    saved = dict(weight_upgraded=np.ones(4) / 4, weight_normalization=np.asarray("unit_area"))
    hist = histogram_bands([0.2, 0.8, 1.2, 1.8], [0, 1, 2], saved)
    np.testing.assert_allclose(hist["central"], [0.5, 0.5])
    np.testing.assert_allclose(hist["sample_error"], [0.25, 0.25])
    full = histogram_bands([0.2, 0.8, 1.2, 1.8], [0, 2], saved, density=False)
    assert full["central"][0] == 1
    assert full["sample_error"][0] == 0
