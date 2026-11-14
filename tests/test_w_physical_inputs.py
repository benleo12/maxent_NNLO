"""Physical frame limits and compatible candidate construction for the W example."""
import json
from pathlib import Path

import numpy as np
import pytest

from examples.w_coordinates import coordinates
from examples.w_physical_inputs import build_blocks, candidate_blocks
from maxent_upgrade import NumericalInput


def decay_events():
    mass = np.tile([50., 80., 150.], 8)
    c = np.linspace(-.8, .8, len(mass))
    phi = np.linspace(.1, 6., len(mass))
    y = np.tile([-.7, .7], len(mass) // 2)
    qt = np.tile([0., 10., 30., 60., 500., 600.], 4)
    momentum = mass / 2
    rest = np.column_stack([momentum * np.sqrt(1 - c**2) * np.cos(phi),
                            momentum * np.sqrt(1 - c**2) * np.sin(phi), momentum * c, momentum])
    opposite = rest.copy()
    opposite[:, :3] *= -1

    def boost(v):
        beta = qt / np.hypot(mass, qt)
        gamma = np.hypot(mass, qt) / mass
        out = v.copy()
        out[:, 0] = gamma * (v[:, 0] + beta * v[:, 3])
        out[:, 3] = gamma * (v[:, 3] + beta * v[:, 0])
        energy = out[:, 3].copy()
        out[:, 2] = np.cosh(y) * v[:, 2] + np.sinh(y) * energy
        out[:, 3] = np.cosh(y) * energy + np.sinh(y) * v[:, 2]
        return out
    lepton, neutrino = boost(rest), boost(opposite)
    return dict(l_born=lepton, nu=neutrino, weight=np.ones(len(mass))), c, y, qt


def test_cartesian_frame_and_continuous_zero_recoil_limit():
    events, c, y, qt = decay_events()
    result = coordinates(events["l_born"], events["nu"])
    np.testing.assert_allclose(result["c"], c * np.where(y < 0, -1, 1), atol=4e-14)
    moving = qt > 0
    np.testing.assert_allclose((result["c"]**2 + result["sx"]**2 + result["sy"]**2)[moving], 1, atol=5e-14)
    np.testing.assert_array_equal(result["rho_sx"][~moving], 0)
    np.testing.assert_array_equal(result["rho_sy"][~moving], 0)
    assert all(np.isfinite(v).all() for v in result.values())


def test_spin_blocks_use_windowed_anchor_and_window_rate_appears_once():
    root = Path(__file__).resolve().parents[1] / "examples/data"
    legacy = json.loads((root / "w_joint/w_wp_training.json").read_text())["accumulator_labels"]
    definitions = json.loads((root / "w_predictivity/feature_definitions.json").read_text())
    labels = legacy + [r["name"] for r in definitions if r["role"] == "fit"]
    mean = np.zeros(len(labels))
    mean[0], mean[labels.index("prof_wptw_0")] = 1000., 200.
    data = NumericalInput(labels, mean, np.eye(len(labels)), "test")
    events, _, _, _ = decay_events()
    for charge in ("wp", "wm"):
        blocks = build_blocks(events, data, charge)
        hard = candidate_blocks(blocks, charge, "H")
        assert hard[2].metadata["transition"] == "windowed"
        with pytest.raises(ValueError, match="anchor R"):
            candidate_blocks(blocks, charge, "anchor+H")
        window = candidate_blocks(blocks, charge, "R+W")
        selected = [label for block in window for label in block.labels]
        assert selected.count("pred_W_0") == 1
        assert len(selected) == len(set(selected))
        assert not any(label.startswith("prof_wctw_") for label in selected)
        assert ("pred_WJ_1_0" in selected) == (charge == "wm")
