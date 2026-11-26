"""Hard-target support certificates and durable diagnostics on the actual prior."""
from copy import deepcopy
import json

import numpy as np
import pytest

from maxent_upgrade import book_targets, fit_targets
from maxent_upgrade.cli import main
from maxent_upgrade.support import InfeasibleTarget, verify_support


def problem(x, y, a=.3, b=-.3):
    events = dict(x=x, y=y, weight=np.ones(len(x)))
    table = dict(config={"born": {k: {"map": "lin", "range": [-1, 1]} for k in ("x", "y")},
                         "moment_selection": False},
                 moments=[dict(observable="x", degree=1, value=a, error=.01),
                          dict(observable="y", degree=1, value=b, error=.01)])
    return events, table


def test_duplicate_observables_reject_conflicting_but_accept_consistent_hard_inputs(tmp_path, capsys):
    x = np.linspace(-1, 1, 3001)
    prior, table = problem(x, x)
    targets, sample, output = [tmp_path / name for name in ("targets.json", "prior.npz", "fit.npz")]
    targets.write_text(json.dumps(table))
    np.savez(sample, **prior)
    with pytest.raises(SystemExit):
        main(["fit", str(sample), str(targets), str(output)])
    message = capsys.readouterr().err
    assert "incompatible hard targets" in message and "x_T1" in message and "y_T1" in message
    assert not output.exists()
    table["moments"][1]["value"] = .3
    fit = fit_targets(prior, table)
    rows = fit.constraint_checks["central"]
    assert rows[1]["name"] == "x_T1" and rows[1]["degree"] == 1
    assert rows[1]["target"] == rows[2]["target"] == .3
    assert rows[1]["achieved"] == rows[2]["achieved"]
    assert abs(rows[1]["achieved"] - .3) < 1e-4


def test_correlated_columns_and_small_affine_violation_even_with_rare_extra_axis():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(5000, 4))
    rare = np.zeros(len(a))
    rare[3] = 1
    x = np.column_stack((a[:, 0], a[:, 0], a, .7*a[:, 2] + .2*a[:, 3] + .4, rare))
    weights = np.exp(.2*a[:, 1]); weights[3] = 250
    goal = np.average(x, axis=0, weights=weights)
    names = np.array([str(i) for i in range(x.shape[1])])
    assert verify_support(x, goal, x.mean(0), names).startswith("in convex hull")
    goal[-2] += 5e-9   # same marginal bounds; no probability distribution can satisfy this
    with pytest.raises(InfeasibleTarget, match="incompatible hard targets"):
        verify_support(x, goal, x.mean(0), names)


def test_separating_facet_and_feasible_target_near_it():
    # Above the slanted side of the triangle, both targets lie within their marginal ranges.
    x, y = np.array([-1, 1, -1]), np.array([-1, -1, 1])
    prior, table = problem(x, y, .25, .25)
    with pytest.raises(InfeasibleTarget, match="largest value"):
        fit_targets(prior, table)
    table["moments"][0]["value"] = 1.
    with pytest.raises(InfeasibleTarget, match="requires zero event weights"):
        fit_targets(prior, table)
    table["moments"][0]["value"] = -.250001
    assert (fit_targets(prior, table).central.weights > 0).all()


def test_soft_targets_and_roundoff_are_not_exact_inconsistencies(tmp_path, capsys):
    x = np.linspace(-1, 1, 205)
    prior, table = problem(x, x)
    table["config"]["soft_constraints"] = True
    fit = fit_targets(prior, table)
    assert fit.constraint_checks["support"] == "no hard shape constraints"
    assert all(r["mode"] == "soft" for r in fit.constraint_checks["central"][1:])
    assert abs(fit.constraint_checks["central"][1]["residual"]) > .29
    path = tmp_path / "soft.npz"
    np.savez(path, **fit.arrays())
    main(["check", str(path)])
    display = capsys.readouterr().out
    assert "Mode" in display and "soft" in display and "hard" in display
    table["moments"][0].update(value=1.05, error=0.5)
    fit_targets(prior, table)
    table["moments"][0].update(value=.3, error=.01)
    # Both coordinates can be softened independently. The joint feasibility check
    # only uses the one with zero supplied error.
    table["moments"][0]["error"] = 0.
    fit_targets(prior, table)
    table["config"]["soft_constraints"] = False
    table["moments"][1]["value"] = .3 + 2e-15
    fit_targets(prior, table)
    # A realizable near-collinear input with correlated coordinates must also pass.
    y = x + .02*np.sin(2*np.pi*x)
    w = np.exp(.5*x)
    nearby, table = problem(x, y, np.average(x, weights=w), np.average(y, weights=w))
    fit_targets(nearby, table)


def test_support_is_checked_in_each_shape_variation_and_rate_only_weights_reuse_shape(tmp_path, capsys):
    x = np.linspace(-1, 1, 100)
    prior, table = problem(x, x, .2, .2)
    table["cross_section_pb"] = 13.
    shifted = deepcopy(table["moments"])
    shifted[1]["value"] = -.15
    table["variations"] = {"scale": [{"label": "incompatible", "moments": shifted}]}
    with pytest.raises(InfeasibleTarget, match="scale member 'incompatible'"):
        fit_targets(prior, table)
    table["variations"] = {"rate": [{"label": "up", "cross_section_pb": 15.}]}
    e = fit_targets(prior, table)
    np.testing.assert_array_equal(e.weights["rate"][0], e.central.weights)
    out = e.arrays()
    assert np.isclose(out["weight_rate"].sum(), 15.)
    checks = json.loads(str(out["constraint_checks_json"]))
    assert checks["variations"]["rate"][0]["constraints"] == checks["central"]
    dest = tmp_path / "result.npz"
    np.savez(dest, **out)
    main(["check", str(dest)])
    assert "x T[1]" in capsys.readouterr().out
    main(["check", str(dest), "--source", "rate", "--member", "up"])
    assert "rate / up" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["check", str(dest), "--source", "rate", "--member", "other"])
    assert "choose --member" in capsys.readouterr().err
    legacy = tmp_path / "old.npz"
    np.savez(legacy, weight_upgraded=e.central.weights)
    with pytest.raises(SystemExit):
        main(["check", str(legacy)])
    assert "no stored per-constraint checks" in capsys.readouterr().err


@pytest.mark.parametrize("transition", ["blend", "windowed"])
def test_window_check_uses_the_actual_fit_feature_and_global_normalization(transition, tmp_path, capsys):
    from maxent_upgrade import profile_w
    pt = np.linspace(.25, 4.12, 999)
    w0 = np.ones_like(pt)
    config = {"recoil": {"pt": {"map": "log", "range": [1, 4], "profile": {"a": 1, "b": 2, "c": 4},
                                "transition": transition}}, "moment_selection": False}
    prior = dict(pt=pt, weight=w0)
    table = book_targets(dict(pt=pt, weight=1 + .2*pt), config, orders=1)
    e = fit_targets(prior, table)
    path = tmp_path / "window.npz"
    np.savez(path, **e.arrays())
    main(["check", str(path)])
    display = capsys.readouterr().out
    assert ("pt <w T[1]>" if transition == "windowed" else "pt T[1] (blend)") in display
    _, record, share = e.constraint_checks["central"]
    c = 2*np.log(np.clip(pt, 1., 4.))/np.log(4.) - 1.
    f = profile_w(pt, 1, 2, 4) if transition == "windowed" else (pt < 4).astype(float)
    assert record["prior"] == pytest.approx(np.mean(f*c))
    assert record["achieved"] == pytest.approx(e.central.weights @ (f*c))
    assert record["residual"] == pytest.approx(record["achieved"] - record["target"])
    assert record["target"] == pytest.approx(float(e.central.report["mu"][1]))
    assert record["input_error"] > 0
    assert share["kind"] == ("window fraction" if transition == "windowed" else "tail fraction")
    del table["moments"][0]["error"], table["moments"][0]["stat_error"]
    e = fit_targets(prior, table)
    assert e.constraint_checks["central"][1]["input_error"] is None
