"""Exact saved-state restoration across numerical reduction environments."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from maxent_upgrade import (
    ConstraintBlock,
    NumericalInput,
    fit_constraints,
    load_frozen,
    plan_observable,
)


def small_fit():
    x = np.linspace(-0.9, 0.9, 300)
    raw = np.exp(0.3 * x)
    data = NumericalInput(("rate", "x"), [10.0, 1.0], [[0.2, 0.01], [0.01, 0.02]], "training")
    block = ConstraintBlock("x", x[:, None], [[0.0, 1.0]], ("x",), metadata=dict(map="linear"))
    return fit_constraints(raw, data, [block], prior_id="events"), raw, x


def test_portable_restore_preserves_exact_arrays_and_identity_guards(tmp_path, monkeypatch):
    fit, raw, x = small_fit()
    plan = {"x2": plan_observable(x * x, labels=("x2",), definition="x squared")}
    path = tmp_path / "portable.npz"
    fit.freeze(path, selection=dict(candidate="x"), validation_plan=plan, portable=True)
    import maxent_upgrade.study as study

    monkeypatch.setattr(
        study, "_newton_maxent", lambda *a, **kw: pytest.fail("reload must not solve")
    )
    restored = load_frozen(path, raw, prior_id="events")
    for field in ("p", "phi", "matrix", "offset", "prior_derivative", "scale_weights", "scale_lam"):
        np.testing.assert_array_equal(getattr(fit, field), getattr(restored, field))
    assert fit.configuration == restored.configuration
    with pytest.raises(ValueError, match="read-only"):
        restored.blocks[0].features[0, 0] += 0.01
    expected = fit.predict(x * x, labels=("x2",), definition="x squared", validation_key="x2")
    actual = restored.predict(x * x, labels=("x2",), definition="x squared", validation_key="x2")
    np.testing.assert_allclose(actual.mean, expected.mean, rtol=1e-14)
    np.testing.assert_allclose(actual.covariance, expected.covariance, rtol=1e-14)
    with pytest.raises(ValueError, match="prior weights"):
        load_frozen(path, raw * 2, prior_id="events")
    with pytest.raises(ValueError, match="event identities"):
        load_frozen(path, raw, prior_id="events", event_ids=np.arange(len(raw))[::-1])
    changed = NumericalInput(fit.data.labels, fit.data.mean + 0.1, fit.data.covariance, "training")
    with pytest.raises(ValueError, match="numerical input"):
        load_frozen(path, raw, changed, prior_id="events")
    altered = ConstraintBlock(
        "x", x[:, None] + 1e-9, [[0.0, 1.0]], ("x",), metadata=dict(map="linear")
    )
    with pytest.raises(ValueError, match="do not match"):
        load_frozen(path, raw, fit.data, [altered], prior_id="events")
    with np.load(path) as source:
        archive = dict(source)
    archive["saved_features"][0, 0] += 1e-9
    tampered = tmp_path / "tampered.npz"
    np.savez(tampered, **archive)
    with pytest.raises(ValueError, match="arrays have changed"):
        load_frozen(tampered, raw, prior_id="events")


def test_portable_freeze_rejects_inconsistent_assembled_features(tmp_path):
    fit, _, x = small_fit()
    plan = {"x2": plan_observable(x * x, labels=("x2",), definition="x squared")}
    fit.phi = np.column_stack([fit.phi, x * x])
    with pytest.raises(ValueError, match="assembled fit dimensions"):
        fit.freeze(
            tmp_path / "extra.npz",
            selection=dict(candidate="x"),
            validation_plan=plan,
            portable=True,
        )
    fit.phi = fit.phi[:, :1].copy()
    fit.phi[0, 0] += 1e-8
    with pytest.raises(ValueError, match="assembled fit arrays"):
        fit.freeze(
            tmp_path / "changed.npz",
            selection=dict(candidate="x"),
            validation_plan=plan,
            portable=True,
        )


def test_portable_reload_in_a_fresh_process_with_different_blas_threads(tmp_path):
    script = tmp_path / "cross_threads.py"
    script.write_text("""import os, sys
from pathlib import Path
import numpy as np
import maxent_upgrade.study as study
from maxent_upgrade import ConstraintBlock, NumericalInput, fit_constraints, load_frozen, plan_observable
root = Path(sys.argv[1])
rng = np.random.default_rng(719)
basis = rng.uniform(-1, 1, (12000, 24))
x = basis[:, 0]
raw = np.exp(.3*x)
h = np.column_stack([x*x, basis[:, 1]])
if sys.argv[2] == "train":
    p = raw/raw.sum()
    part = (p@basis)[0]*.1
    data = NumericalInput(("rate", "x"), [10., .5], [[.2, .01], [.01, .02]], "training")
    block = ConstraintBlock("blend", x[:, None], [[0., 1.]], ("x",), [part], .1*x[:, None],
                            metadata=dict(definition="linear prior-dependent offset"))
    fit = fit_constraints(raw, data, [block], prior_id="events")
    fit.freeze(root/"fit.npz", selection=dict(candidate="blend"), portable=True,
               validation_plan={"h": plan_observable(h, labels=("x2", "y"), definition="declared columns")})
else:
    def no_solve(*a, **kw): raise AssertionError("reload attempted a solve")
    study._newton_maxent = no_solve
    fit = load_frozen(root/"fit.npz", raw, prior_id="events")
result = fit.predict(h, labels=("x2", "y"), definition="declared columns", validation_key="h")
np.savez(root/(sys.argv[2]+".npz"), weights=fit.scale_weights, offset=fit.offset, mean=result.mean,
         covariance=result.covariance, requested_threads=int(os.environ["OPENBLAS_NUM_THREADS"]))
""")
    repo = str(Path(__file__).resolve().parents[1])
    for stage, threads in (("train", "1"), ("reload", "2")):
        environment = dict(os.environ, PYTHONPATH=repo)
        environment.update(
            {
                key: threads
                for key in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            }
        )
        subprocess.run(
            [sys.executable, str(script), str(tmp_path), stage],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    with np.load(tmp_path / "train.npz") as first, np.load(tmp_path / "reload.npz") as second:
        assert first["requested_threads"] == 1 and second["requested_threads"] == 2
        np.testing.assert_array_equal(first["weights"], second["weights"])
        np.testing.assert_array_equal(first["offset"], second["offset"])
        np.testing.assert_allclose(first["mean"], second["mean"], rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(
            first["covariance"], second["covariance"], rtol=2e-12, atol=2e-13
        )
