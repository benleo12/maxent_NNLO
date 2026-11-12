"""User-facing workflows: custom observables, CLI files, normalization and errors."""
from pathlib import Path
import runpy

import numpy as np
import pytest

from maxent_upgrade import compute_fo_moments, load_moments, released, upgrade
from maxent_upgrade.cli import main


def custom_problem():
    x = np.linspace(-1.0, 1.0, 512)
    events = dict(my_observable=x, weight=1.0 + 0.15 * x, event_id=np.arange(len(x)))
    config = {"born": {"my_observable": {"range": (-1.0, 1.0), "map": "lin"}}}
    fixed_order = dict(events, weight=events["weight"] * np.exp(0.4 * x))
    return events, compute_fo_moments(fixed_order, config, n_max=4), config


def test_custom_observable_recovers_known_correction():
    events, moments, config = custom_problem()
    result = upgrade(events, moments, config)
    target = events["weight"] * np.exp(0.4 * events["my_observable"])
    target /= target.sum()
    assert (result.weights > 0).all()
    assert abs(result.weights.sum() - 1.0) < 1e-12
    assert np.max(abs(result.weights / target - 1)) < 2e-4
    assert "my_observable" in result.describe()
    assert "Weights sum to one" in result.describe()


def test_cli_preserves_events_and_uses_cross_section_weights(tmp_path, monkeypatch):
    events, moments, config = custom_problem()
    events["cross_section_pb"] = np.asarray(4.0)  # prior metadata must also survive
    monkeypatch.setattr(released, "load_moments", lambda *a, **k: (moments, config))
    monkeypatch.setattr(released, "table", lambda *a: {"fiducial": "synthetic selection"})
    monkeypatch.setattr(released, "sigma_fid", lambda *a: {"value": 12.5})
    prior = tmp_path / "prior.npz"
    posterior = tmp_path / "posterior.npz"
    np.savez(prior, **events)
    main(["reweight", "custom", str(prior), str(posterior)])
    with np.load(posterior) as saved:
        for name, values in events.items():
            assert np.array_equal(saved[name], values)
        assert np.isclose(saved["weight_upgraded"].sum(), 12.5)
        assert float(saved["cross_section_upgraded_pb"]) == 12.5
        expected = upgrade(events, moments, config).weights * 12.5
        np.testing.assert_allclose(saved["weight_upgraded"], expected)
    before = posterior.read_bytes()
    with pytest.raises(SystemExit):
        main(["reweight", "custom", str(prior), str(posterior)])
    assert posterior.read_bytes() == before
    main(["reweight", "custom", str(prior), str(posterior), "--force", "--unit-area"])
    with np.load(posterior) as saved:
        assert np.isclose(saved["weight_upgraded"].sum(), 1)
        assert "cross_section_upgraded_pb" not in saved
        assert float(saved["cross_section_pb"]) == 4.0


@pytest.mark.parametrize("weights", [np.empty(0), np.ones((2, 3)), np.array(1.0)])
def test_invalid_event_shapes_fail_before_solving(weights):
    events, moments, config = custom_problem()
    events["weight"] = weights
    with pytest.raises(ValueError, match="one-dimensional|empty"):
        upgrade(events, moments, config)


def test_inspect_uses_selected_higgs_window(capsys):
    main(["inspect", "ggh"])
    output = capsys.readouterr().out
    assert "54.0" in output and "108.0" in output
    assert "pT_H" in output and "Selection:" in output


def test_all_paper_configurations_load():
    script = Path(__file__).resolve().parents[1] / "examples" / "paper_examples.py"
    examples = runpy.run_path(str(script))["EXAMPLES"]
    for process, options, _ in examples.values():
        # Parsing the actual command also checks empty --mixed and added windows.
        main(["inspect", process, *options])
    assert {"wp-angle", "wm-angle", "zj-no-joint", "zj-acoplanarity"} <= examples.keys()


def test_variation_cross_sections_follow_released_accumulators():
    for process in released.processes():
        central = released.sigma_fid(process)["value"]
        assert released.cross_section(process) == central
        columns, accumulators, _ = released._runs(released.PROCESSES[process])
        rates = accumulators[:, columns["norm_born"], :]
        for scale in range(7):
            expected = central * rates[:, scale].mean() / rates[:, 0].mean()
            assert np.isclose(released.cross_section(process, scale=scale), expected)
        runs = [0, 0, 1]
        expected = central * rates[runs, 0].mean() / rates[:, 0].mean()
        assert np.isclose(released.cross_section(process, runs=runs), expected)
    for process in ("wp", "wm", "ggh"):
        table = released.pdf_table(process)
        for member in (0, 1, 29, 58):
            expected = released.sigma_fid(process)["value"] * table["sigma_fid_ratio"][member]
            assert released.cross_section(process, pdf_member=member) == expected


def test_invalid_variation_cross_sections():
    for kwargs in ({"scale": 7}, {"runs": []}, {"runs": [-1]}, {"pdf_member": 1}):
        with pytest.raises(ValueError):
            released.cross_section("dy", **kwargs)


def test_custom_joint_example_runs():
    script = Path(__file__).resolve().parents[1] / "examples" / "custom_observables.py"
    sample = runpy.run_path(str(script))["run"](1000)
    product = (sample["mass"] - 100.0) / 40.0 * sample["cos_theta"]
    target = sample["weight_target"] @ product
    joint_error = abs(sample["weight_upgraded"] @ product - target)
    separate_error = abs(sample["weight_separate"] @ product - target)
    assert joint_error < 1e-4
    assert separate_error > 10 * joint_error


@pytest.mark.parametrize("n_boot", [0, 2])
def test_band_example_saves_rates_and_uses_cold_window_starts(tmp_path, monkeypatch, n_boot):
    """Exercise the band driver on a tiny problem, keeping the real released rates."""
    import sys

    events, moments, config = custom_problem()
    prior, output = tmp_path / "prior.npz", tmp_path / "bands.npz"
    np.savez(prior, **events)
    script = Path(__file__).resolve().parents[1] / "examples" / "bands_with_released_moments.py"
    main = runpy.run_path(str(script))["main"]
    calls = []

    def load_small_problem(process, **kwargs):
        calls.append(kwargs)
        return moments, config

    solves = []

    def solve_small_problem(events, moments, config):
        solves.append(config)
        return upgrade(events, moments, config)

    members = dict(released.pdf_table("wp"), n_members=3)
    monkeypatch.setattr(released, "pdf_table", lambda process: members)
    monkeypatch.setitem(main.__globals__, "load_moments", load_small_problem)
    monkeypatch.setitem(main.__globals__, "upgrade", solve_small_problem)
    monkeypatch.setattr(sys, "argv", [str(script), "wp", str(prior), str(n_boot), str(output)])
    main()
    for call, solve in zip(calls[1:], solves[1:]):
        assert "fixed_orders" in solve
        assert ("lam0" not in solve) == ("window" in call)
    with np.load(output) as saved:
        assert saved["scale_w"].shape == (6, len(events["weight"]))
        assert saved["boot_w"].shape == (n_boot, len(events["weight"]))
        assert saved["pdf_w"].shape == (3, len(events["weight"]))
        assert np.allclose(saved["scale_w"].sum(axis=1), 1)
        assert np.allclose(saved["sigma_scale_pb"],
                           [released.cross_section("wp", scale=k) for k in range(1, 7)])
        assert np.allclose(saved["sigma_pdf_pb"],
                           [released.cross_section("wp", pdf_member=k) for k in range(3)])
        replicas = [call["runs"] for call in calls if "runs" in call]
        assert np.allclose(saved["sigma_boot_pb"],
                           [released.cross_section("wp", runs=runs) for runs in replicas])
        for name in ("central", "window_down", "window_up"):
            assert np.isclose(saved[name].sum(), 1)
        assert saved["sigma_window_pb"] == saved["sigma_central_pb"]
