import json

import numpy as np
import pytest

from maxent_upgrade.cli import main


def test_selection_report_includes_skipped_degrees_and_explicit_products(tmp_path, capsys):
    x = np.tile([-1., 0., 1.], 12)
    y = np.repeat([-1., 0., 1.], 12)
    prior = tmp_path / "prior.npz"
    np.savez(prior, weight=np.ones_like(x), x=x, y=y)
    table = {
        "config": {
            "born": {"x": {"range": [-1, 1], "map": "lin"}},
            "mixed": {"xy": {"observables": ["x", "y"], "n": 1,
                             "range": {"y": [-1, 1]}, "map": {"y": "lin"}}}},
        "moments": [
            {"observable": "x", "degree": 1, "value": .1, "error": 0.},
            {"observable": "x", "degree": 4, "value": 1., "error": 0.},
            {"observable": "xy", "degree": [1, 1], "value": 0., "error": .2}]}
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps(table))
    output = tmp_path / "output.npz"
    main(["fit", str(prior), str(targets), str(output)])
    main(["check", str(output), "--selection"])
    display = capsys.readouterr().out
    assert "inf | yes" in display and "0 | no" in display and "supplied joint moment" in display
    with np.load(output) as fitted:
        rows = json.loads(str(fitted["constraint_checks_json"]))["prefit"]
        assert [(r["degree"], r["retained"]) for r in rows] == [(1, True), (4, False), ([1, 1], True)]
        assert rows[-1]["snr"] == pytest.approx(0., abs=1e-12)


def test_invalid_prior_rejected_before_native_calculation(tmp_path, monkeypatch, capsys):
    from maxent_upgrade import nnlojet

    def unexpected(*args, **kwargs):
        pytest.fail("started an integration with invalid prior weights")

    monkeypatch.setattr(nnlojet, "calculate", unexpected)
    setup = tmp_path / "calc.json"
    setup.write_text(json.dumps({"runcard": "test.run", "axes": {"x": "pt"},
                                 "config": {"born": {"x": {"range": [0, 1], "map": "lin"}}}}))
    prior = tmp_path / "prior.npz"
    np.savez(prior, weight=[1., -1.], x=[0., 1.])
    dest = tmp_path / "output.npz"
    with pytest.raises(SystemExit):
        main(["match", str(prior), str(setup), str(dest)])
    assert "strictly positive" in capsys.readouterr().err
    assert not dest.exists()
