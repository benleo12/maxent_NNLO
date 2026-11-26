"""The portable target table is the user-facing definition of each fitted moment."""
from copy import deepcopy
import json

import numpy as np
import pytest

from maxent_upgrade import book_targets, fit_targets
from maxent_upgrade.cli import main


def inputs():
    x = np.linspace(-1, 1, 400)
    y = np.sin(14 * x)
    # Independent integration records, including signed contributions.
    w = np.where(np.arange(len(x)) % 4 == 0, -0.2, 1.0) * np.exp(0.3 * x + 0.2 * x * y)
    prior = dict(weight=1 + x[::2] / 10, x=x[::2], y=y[::2], event_id=np.arange(0, len(x), 2))
    fo = dict(x=x, y=y, weight=w, weight_scales=np.stack([w, w * (1.1 + 0.05 * x), 0.9 * w]),
              scale_labels=np.array(["central", "up", "down"]))
    config = {"born": {"x": {"range": [-1, 1], "map": "lin"}},
              "mixed": {"xy": {"observables": ["x", "y"], "range": {"x": [-1, 1], "y": [-1, 1]},
                               "map": {"x": "lin", "y": "lin"}, "n": 2}}, "moment_selection": False}
    return prior, fo, config


def test_signed_records_to_portable_targets_to_ordered_weights(tmp_path):
    prior, fo, config = inputs()
    cfg, sample, targets, showers, dest = [tmp_path / name for name in (
        "maps.json", "integration.npz", "targets.json", "prior.npz", "weighted.npz")]
    cfg.write_text(json.dumps(config))
    np.savez(sample, **fo)
    np.savez(showers, **prior)
    main(["moments", str(cfg), str(sample), str(targets), "--orders", "2", "--cross-section", "12.4",
          "--selection", "fiducial example"])
    table = json.loads(targets.read_text())
    assert table["selection"] == "fiducial example"
    table["moments"].reverse()
    for member in table["variations"]["scale"]:
        member["moments"].reverse()
    targets.write_text(json.dumps(table))
    main(["fit", str(showers), str(targets), str(dest)])
    with np.load(dest) as d:
        for name in prior:
            np.testing.assert_array_equal(d[name], prior[name])
        assert d["weight_scale"].shape == (2, len(prior["x"]))
        np.testing.assert_array_equal(d["labels_scale"], ["up", "down"])
        np.testing.assert_allclose(d["weight_scale"].sum(axis=1), 12.4 * fo["weight_scales"][1:].sum(axis=1) / fo["weight"].sum())
        assert d["weight_normalization"] == "pb"
        assert np.isclose(d["weight_upgraded"].sum(), 12.4)
        for original, result in zip(fo["weight_scales"], [d["weight_upgraded"], *d["weight_scale"]]):
            assert (result > 0).all()
            for f in (lambda x, y: x, lambda x, y: 2*x*x-1, lambda x, y: x*(2*y*y-1)):
                assert np.isclose(np.average(f(fo["x"], fo["y"]), weights=original),
                                  np.average(f(prior["x"], prior["y"]), weights=result), atol=1e-4)
    old = dest.read_bytes()
    with pytest.raises(SystemExit):
        main(["fit", str(showers), str(targets), str(dest)])
    assert dest.read_bytes() == old


def test_fixed_degrees_without_synthetic_error_or_cross_section():
    prior, fo, config = inputs()
    table = book_targets(fo, config, orders=2)
    for record in [table, *table["variations"]["scale"]]:
        for row in record["moments"]:
            del row["error"], row["stat_error"]
    fit = fit_targets(prior, table)
    out = fit.arrays()
    assert fit.central.chosen_moments == {"x": 2, "xy": 4}
    assert np.isnan(fit.central.max_pull)
    assert out["weight_normalization"] == "unit_area"
    assert "cross_section_upgraded_pb" not in out
    assert json.loads(str(out["provenance_json"]))["target_cross_section_pb"] is None
    np.testing.assert_allclose(out["weight_scale"].sum(axis=1), 1.)
    table["config"]["moment_selection"] = True
    with pytest.raises(ValueError, match="supply error"):
        fit_targets(prior, table)


@pytest.mark.parametrize("change,problem", [
    (lambda t: t["moments"][0].update(observable="unknown"), "undefined observable"),
    (lambda t: t["moments"].append(t["moments"][0]), "duplicate moment"),
    (lambda t: t["moments"][0].update(degree=0), "positive integer"),
    (lambda t: t["moments"][0].update(value=2), "outside prior support"),
    (lambda t: t["moments"][0].update(error=-1), "out of range"),
    (lambda t: t["config"]["born"]["x"].update(map="bw"), "pole mass"),
    (lambda t: t["config"]["born"]["x"].update(units_typo=True), "unrecognized"),
    (lambda t: t["variations"]["scale"][0]["moments"].pop(), "match the central"),
    (lambda t: t["variations"]["scale"][0].update(config={"born": {}}), "unrecognized variation"),
])
def test_reject_ambiguous_or_unmatched_inputs(change, problem):
    prior, fo, config = inputs()
    table = book_targets(fo, config, orders=2)
    change(table)
    with pytest.raises(ValueError, match=problem):
        fit_targets(prior, table)


def test_window_targets_and_alternate_prior_weights():
    prior, fo, _ = inputs()
    config = {"weight_key": "nominal", "moment_selection": False,
              "recoil": {"pt": {"range": [1, 4], "map": "log",
                                "profile": {"a": 1, "b": 2, "c": 3}, "transition": "windowed"}}}
    fo = dict(pt=2 + 2*fo["x"], nominal=fo["weight"])
    prior = dict(pt=2 + 2*prior["x"], nominal=prior["weight"], scale_up=1 + 0.7 * prior["x"])
    targets = book_targets(fo, config, orders=2)
    targets["variations"] = {"shower": [{"label": "up", "prior_weight_column": "scale_up"}]}
    fit = fit_targets(prior, targets)
    assert fit.weights["shower"].shape == (1, len(prior["nominal"]))
    assert not np.allclose(fit.central.weights, fit.weights["shower"][0])
    no_rates = deepcopy(targets)
    del no_rates["rates"]
    with pytest.raises(ValueError, match="window fraction"):
        fit_targets(prior, no_rates)
    with pytest.raises(ValueError, match="signed weight sum"):
        book_targets(dict(fo, nominal=-fo["nominal"]), config, orders=2)
    with pytest.raises(ValueError, match="signed weight sum"):
        book_targets(dict(fo, pt=np.zeros_like(fo["pt"])), config, orders=2)
    with pytest.raises(ValueError, match="one finite value"):
        book_targets(dict(fo, pt=fo["pt"][:-1]), config)
    with pytest.raises(ValueError, match="row zero"):
        book_targets(dict(fo, weight_scales=[-fo["nominal"]]), config)


def test_public_two_moment_example_matches_released_tables():
    from pathlib import Path
    from maxent_upgrade import released
    table = json.loads((Path(__file__).resolve().parents[1] / "examples/own_prior/targets.json").read_text())
    for i, entry in enumerate([table, *table["variations"]["scale"]]):
        target, _ = released.load_moments("dy", born=["y_abs"], recoil=[], mixed=[], scale=i if i else None)
        assert entry["cross_section_pb"] == released.cross_section("dy", scale=i if i else None)
        for row in entry["moments"]:
            assert row["value"] == target["born"]["y_abs"]["values"][row["degree"] - 1]

def test_joint_axis_map_reused_or_checked_by_id():
    prior, fo, config = inputs()
    table = book_targets(fo, config, orders=2)
    baseline = fit_targets(prior, table).central.weights
    for field in ("map", "range"):
        del table["config"]["mixed"]["xy"][field]["x"]
    np.testing.assert_array_equal(baseline, fit_targets(prior, table).central.weights)
    table["config"]["mixed"]["xy"]["map"]["x"] = ["bw", 1.0, 0.1]
    with pytest.raises(ValueError, match="conflicting map definitions"):
        fit_targets(prior, table)


@pytest.mark.parametrize("select", [False, True])
def test_sparse_even_degrees_leave_odd_angular_moment_free(select):
    x = np.linspace(-1, 1, 1001)
    prior = dict(angle=x, weight=1.0 + 0.6*x)
    t2, t4 = 2*x*x - 1, 8*x**4 - 8*x*x + 1
    target_weights = prior["weight"] * np.exp(0.35*t2 - 0.15*t4)
    targets = {n: np.average(t, weights=target_weights) for n, t in [(2, t2), (4, t4)]}
    table = dict(config={"born": {"angle": {"range": [-1, 1], "map": "lin"}},
                         "moment_selection": select, "L2": 0.},
                 moments=[dict(observable="angle", degree=d, value=targets[d],
                               error=1.0 if select and d == 4 else .0001) for d in (4, 2)])
    changed = target_weights * np.exp(.1*t2)
    table["variations"] = {"scale": [dict(label="shift", moments=[
        dict(observable="angle", degree=d, value=np.average(t, weights=changed), error=.00001)
        for d, t in [(2, t2), (4, t4)]])]}
    e = fit_targets(prior, table)
    expected = [2] if select else [2, 4]
    assert e.central.report["fitted_degrees"] == {"angle": expected}
    assert e.central.report["feature_names"] == ["norm"] + [f"angle_T{d}" for d in expected]
    assert e.central.chosen_moments == {"angle": len(expected)}
    assert json.loads(str(e.arrays()["moment_degrees_json"])) == {"angle": expected}
    for fit_weights in (e.central.weights, e.weights["scale"][0]):
        assert np.isclose(fit_weights @ t2, np.average(t2, weights=target_weights if fit_weights is e.central.weights else changed))
        np.testing.assert_allclose(fit_weights / fit_weights[::-1], prior["weight"]/prior["weight"][::-1], rtol=1e-12)
    assert abs(e.central.weights @ x - np.average(x, weights=prior["weight"])) > .01
    if not select:
        np.testing.assert_allclose(e.central.weights, target_weights / target_weights.sum(), atol=1e-11)
        assert np.isclose(e.weights["scale"][0] @ t4, np.average(t4, weights=changed))
    else:
        assert e.central.report["towers"][0]["fitted_degrees"] != [2, 4]
        # The very precise fourth-degree variation did not change the central constraint set.
        assert abs(e.weights["scale"][0] @ t4 - np.average(t4, weights=changed)) > .01


@pytest.mark.parametrize("transition", ["windowed", "blend"])
def test_sparse_recoil_only_imposes_named_polynomial_in_the_window(transition):
    pt = np.linspace(0.1, 4.0, 600)
    prior = dict(pt=pt, weight=np.ones_like(pt))
    config = dict(recoil={"pt": {"range": [1, 4], "map": "log", "transition": transition,
                                 "profile": {"a": 1, "b": 2, "c": 4}}},
                  moment_selection=False, L2=0.)
    table = book_targets(dict(pt=pt, weight=1 + .15*pt), config, orders=3)
    table["moments"] = [row for row in table["moments"] if row["degree"] == 3]
    fit = fit_targets(prior, table).central
    assert fit.report["fitted_degrees"] == {"pt": [3]}
    assert "pt_T3" in fit.report["feature_names"]
    assert "pt_T1" not in fit.report["feature_names"]
    assert fit.max_pull < 1e-6
