"""Training halves preserve common integrals and physical run identities."""
import json

import numpy as np
import pytest

from examples.w_physical_inputs import load_input
from examples.w_selection import load_training


def write_input(path, *, group="T", invalid_seed=False):
    seeds = np.array([[14, 10, 16, 12, 18, 20], [25, 21, 27, 23, -1, -1]])
    runs = np.full((2, 6, 3, 1), np.nan)
    for channel in range(2):
        actual = seeds[channel] >= 0
        s = seeds[channel, actual].astype(float)
        runs[channel, actual, :, 0] = np.column_stack([100 + s, s * s, 2 * s + 1])
    groups = [a[s >= 0] for a, s in zip(runs, seeds)]
    if invalid_seed:
        seeds[-1, -1] = -2
    np.savez(path, acc=runs, seeds=seeds,
             mean=sum(a.mean(axis=0) for a in groups),
             cov=sum(np.cov(a[:, :, 0], rowvar=False) / len(a) for a in groups),
             metadata=np.array(json.dumps(dict(group=group, charge="WP", labels=["rate", "a", "b"]))))
    return runs, seeds


def test_training_halves_use_sorted_seeds_and_complete_channel_covariance(tmp_path):
    path = tmp_path / "training.npz"
    runs, seeds = write_input(path)
    halves = [load_training(path, name) for name in ("even", "odd")]
    for parity, result in enumerate(halves):
        selected = []
        for values, ids in zip(runs, seeds):
            actual = np.flatnonzero(ids >= 0)
            take = actual[np.argsort(ids[actual])][parity::2]
            selected.append(values[take])
        np.testing.assert_allclose(result.mean, sum(a.mean(axis=0) for a in selected))
        np.testing.assert_allclose(result.covariance,
                                   sum(np.cov(a[:, :, 0], rowvar=False) / len(a) for a in selected))
        assert abs(result.covariance[0, 1]) > 0  # normalization/observable correlation survives
    assert set(halves[0].run_ids).isdisjoint(halves[1].run_ids)
    assert set(halves[0].run_ids) | set(halves[1].run_ids) == set(load_training(path).run_ids)


def test_training_rejects_reference_role_and_invalid_seed_markers(tmp_path):
    path = tmp_path / "input.npz"
    write_input(path, group="V")
    with pytest.raises(ValueError, match="refuses validation"):
        load_training(path)
    write_input(path, group="unknown")
    with pytest.raises(ValueError, match="group must be"):
        load_input(path)
    write_input(path, invalid_seed=True)
    with pytest.raises(ValueError, match="-1 padding"):
        load_input(path)


def test_w_workflow_rejects_cross_charge_inputs(tmp_path):
    from examples.w_constraint_workflow import check_input_charge, load_events
    path = tmp_path / "input.npz"
    write_input(path)
    check_input_charge(load_input(path), "wp")
    with pytest.raises(ValueError, match="requested W charge"):
        check_input_charge(load_input(path), "wm")
    np.savez(path, event_id=np.array([1, 2]), charge=np.array([1, -1]))
    with pytest.raises(ValueError, match="only the requested W charge"):
        load_events(path, "wp")
