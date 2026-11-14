"""Saved results must identify the actual fit, input data and normalization."""
import json

import numpy as np

from maxent_upgrade import Variation, compute_fo_moments, fit_provenance, upgrade, upgrade_ensemble
from maxent_upgrade.provenance import array_identity, canonical_json, content_hash, table_identity


def problem():
    x = np.linspace(-1, 1, 100)
    events = dict(x=x, weight=np.ones(100), event_id=np.arange(100))
    config = {"born": {"x": dict(map="lin", range=(-1, 1))}}
    moments = compute_fo_moments(dict(events, weight=np.exp(.3*x)), config, n_max=3)
    return events, moments, config


def test_hashes_are_order_independent_and_change_with_actual_inputs():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert content_hash({(1, 2): np.float64(.4)}) == content_hash({(1, 2): .4})
    assert array_identity(np.arange(3)) != array_identity(np.arange(3)[::-1])
    assert array_identity(np.arange(3)) != array_identity(np.arange(3).astype(float))


def test_full_fit_record_identifies_data_and_weight_units(tmp_path):
    events, moments, config = problem()
    result = upgrade(events, moments, config)
    path = tmp_path / "prior.npz"
    np.savez(path, **events)
    record = fit_provenance(events, moments, config, result, cross_section=12,
                            input_path=path, table=table_identity("custom", {"order": "NNLO"}))
    record = json.loads(canonical_json(record))
    assert record["configuration"]["L2"] == 1e-4
    assert record["configuration"]["born"]["x"]["range"] == [-1, 1]
    assert record["event_arrays"]["event_id"] == array_identity(events["event_id"])
    assert record["output_weights"] == array_identity(result.weights * 12)
    assert record["moment_table"]["order"] == "NNLO"
    assert record["input_file"]["name"] == "prior.npz"
    assert len(record["input_file"]["sha256"]) == 64


def test_ensemble_records_each_changed_target_and_normalization():
    events, moments, config = problem()
    changed = compute_fo_moments(dict(events, weight=np.exp(.6*events["x"])), config, n_max=3)
    result = upgrade_ensemble(events, moments, config,
                              {"scale": [Variation(changed, cross_section=15)]}, cross_section=12)
    absolute = json.loads(str(result.arrays()["provenance_json"]))
    normalized = json.loads(str(result.arrays(unit_area=True)["provenance_json"]))
    assert absolute["normalization"] == "pb" and normalized["normalization"] == "unit_area"
    assert absolute["target_cross_section_pb"] == normalized["target_cross_section_pb"] == 12
    shift = absolute["variations"]["scale"][0]
    assert shift["moments_sha256"] != absolute["moments_sha256"]
    assert shift["cross_section_pb"] == 15
    assert shift["configuration"]["fixed_orders"] == result.central.chosen_moments
    assert normalized["output_weights"] == array_identity(result.central.weights)
