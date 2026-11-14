"""Optional record extraction, real fit and ID-based event-weight round trip."""
import json

import numpy as np
import pytest

hepmc = pytest.importorskip("pyhepmc")
from maxent_upgrade import compute_fo_moments, upgrade
from maxent_upgrade.hepmc import attach_weights, collect_dy, dy_observables, extract_dy
from maxent_upgrade.provenance import file_hash


def record(number, mass=91., costh=.2, mev=False):
    run = hepmc.GenRunInfo()
    run.weight_names = ["nominal", "variation"]
    run.tools = [("fixture", "1", "deterministic unit-test event")]
    run.attributes["original_run_attribute"] = "retained"
    run.attributes["original_integer"] = 17
    run.attributes["original_real"] = 3.125
    run.attributes["original_vector"] = [1., -2.5, 3.]
    event = hepmc.GenEvent(hepmc.Units.MEV if mev else hepmc.Units.GEV, hepmc.Units.MM)
    event.run_info = run
    event.event_number = number
    event.weights = [1+number*.001, 2.]
    event.attributes["original_event_attribute"] = "retained"
    vertex = hepmc.GenVertex()
    factor = 1000 if mev else 1
    energy = mass*factor/2
    beam1 = hepmc.GenParticle((0, 0, 6500*factor, 6500*factor), 2212, 4)
    beam2 = hepmc.GenParticle((0, 0, -6500*factor, 6500*factor), 2212, 4)
    vertex.add_particle_in(beam1)
    vertex.add_particle_in(beam2)
    px, pz = energy*np.sqrt(1-costh**2), energy*costh
    vertex.add_particle_out(hepmc.GenParticle((px, 0, pz, energy), 11, 1))
    vertex.add_particle_out(hepmc.GenParticle((-px, 0, -pz, energy), -11, 1))
    event.add_vertex(vertex)
    return event


def write_records(path, records):
    with hepmc.open(path, "w") as stream:
        for event in records:
            stream.write(event)


def test_extract_units_ancestry_cuts_and_stream_counters(tmp_path):
    event = record(10, mev=True)
    # A secondary e+e- pair from a hadron must not be chosen as the signal.
    decay = hepmc.GenVertex()
    decay.add_particle_in(hepmc.GenParticle((0, 0, 0, 1000), 111, 2))
    decay.add_particle_out(hepmc.GenParticle((300, 0, 0, 500), 11, 1))
    decay.add_particle_out(hepmc.GenParticle((-300, 0, 0, 500), -11, 1))
    event.add_vertex(decay)
    result = dy_observables(event, qed_off=True)
    assert result["mll"] == pytest.approx(91)
    assert result["costh_cs"] == pytest.approx(.2)
    assert result["phistar"] == pytest.approx(0, abs=1e-12)
    assert event.momentum_unit == hepmc.Units.MEV
    assert dy_observables(record(11, mass=150), qed_off=True) is None
    with pytest.raises(ValueError, match="pre-QED"):
        dy_observables(event)
    arrays = collect_dy(iter([event, record(11, mass=150)]), qed_off=True)
    metadata = json.loads(str(arrays["record_provenance_json"]))
    assert metadata["source_records"] == 2 and metadata["selected_records"] == 1
    assert metadata["source_weight_sum"] == pytest.approx(2.021)
    assert arrays["event_id"].tolist() == [10]


def test_record_arrays_fit_and_permuted_weight_attachment(tmp_path):
    source, output = tmp_path/"input.hepmc", tmp_path/"weighted.hepmc"
    write_records(source, [record(i, mass=75+30*i/79) for i in range(80)])
    arrays = extract_dy(source, qed_off=True)
    config = {"born": {"mll": {"range": (66, 116), "map": "lin"}}, "L2": 0, "fixed_orders": {"mll": 3}}
    target = dict(arrays, weight=arrays["weight"]*np.exp(.01*(arrays["mll"]-90)))
    moments = compute_fo_moments(target, config, n_max=3)
    result = upgrade(arrays, moments, config)
    assert not np.allclose(result.weights, arrays["weight"]/arrays["weight"].sum())
    weights = 12*result.weights
    permutation = np.arange(79, -1, -1)
    variations = {"maxent_scale_001_pb_per_event": weights*1.1,
                  "maxent_stat_001_pb_per_event": weights*np.linspace(.9, 1.2, len(weights))}
    metadata = attach_weights(source, output, arrays["event_id"][permutation], weights[permutation],
                              expected_source_sha256=file_hash(source), provenance={"test": "actual fit"},
                              variations_pb={key: vector[permutation] for key, vector in variations.items()})
    assert metadata["cross_section_pb"] == pytest.approx(12)
    assert metadata["variation_cross_sections_pb"]["maxent_scale_001_pb_per_event"] == pytest.approx(13.2)
    with hepmc.open(source) as before, hepmc.open(output) as after:
        for old, new in zip(before, after):
            assert new.event_number == old.event_number
            assert list(new.weights[:-3]) == list(old.weights)
            assert new.weights[-3] == pytest.approx(weights[new.event_number])
            for offset, vector in enumerate(variations.values()):
                assert new.weights[-2+offset] == pytest.approx(vector[new.event_number])
            assert new.run_info.weight_names == ["nominal", "variation", "maxent_pb_per_event", *variations]
            assert str(new.attributes["original_event_attribute"]) == "retained"
            assert str(new.run_info.attributes["original_run_attribute"]) == "retained"
            for key, kind, expected in (("original_integer", int, 17), ("original_real", float, 3.125),
                                        ("original_vector", list[float], [1., -2.5, 3.])):
                value = new.run_info.attributes[key]
                actual = value.astype(kind) if hasattr(value, "astype") else value
                assert actual == expected
            assert [(p.pid, p.status, tuple(p.momentum)) for p in old.particles] == [
                (p.pid, p.status, tuple(p.momentum)) for p in new.particles]
    restored = extract_dy(output, qed_off=True)
    np.testing.assert_array_equal(restored["event_id"], arrays["event_id"])
    np.testing.assert_array_equal(restored["weight"], arrays["weight"])


def test_attachment_rejects_wrong_source_missing_or_duplicate_ids_atomically(tmp_path):
    source, output = tmp_path/"input.hepmc", tmp_path/"weighted.hepmc"
    write_records(source, [record(1), record(2)])
    for ids, expected in (([1, 1], "unique"), ([1, 99], "absent")):
        with pytest.raises(ValueError, match=expected):
            attach_weights(source, output, np.array(ids), np.ones(2))
        assert not output.exists()
    with pytest.raises(ValueError, match="hash"):
        attach_weights(source, output, np.array([1]), np.ones(1), expected_source_sha256="wrong")
    assert not output.exists()
    with pytest.raises(ValueError, match="duplicate"):
        collect_dy([record(1), record(1)], qed_off=True)


def test_selected_records_only_and_stream_subset_limit(tmp_path):
    subset = tmp_path/"subset.hepmc"
    with hepmc.open(subset, "w") as writer:
        arrays = collect_dy([record(1, mass=150), record(2), record(3)], qed_off=True,
                            selected_writer=writer, selected_limit=1)
    assert arrays["event_id"].tolist() == [2, 3]
    with hepmc.open(subset) as reader:
        assert [event.event_number for event in reader] == [2]
    output = tmp_path/"weighted.hepmc"
    attach_weights(subset, output, np.array([2]), np.array([7.]))
    with hepmc.open(output) as reader:
        records = list(reader)
    assert len(records) == 1 and records[0].weights[-1] == 7
