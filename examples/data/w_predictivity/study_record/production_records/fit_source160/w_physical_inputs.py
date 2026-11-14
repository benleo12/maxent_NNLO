"""Declared W physical blocks for the generic common-covariance workflow.

This module constructs measurement functions and target selectors. It performs
no model selection and never reads a validation reference. The study and public
demonstration share this builder; no second fitting implementation is needed.
"""
import json
from pathlib import Path

import numpy as np

from maxent_upgrade import ConstraintBlock, NumericalInput, profile_w
from maxent_upgrade.basis import cheb, umap
from maxent_upgrade.provenance import file_hash
if __package__:
    from .w_coordinates import coordinates
else:
    from w_coordinates import coordinates


DATA = Path(__file__).resolve().parent / "data/w_predictivity"
ORDERS = {"wp": dict(m_w=12, y_abs=12, pt_w=19, joint=0),
          "wm": dict(m_w=9, y_abs=12, pt_w=20, joint=2)}
WINDOWS = {"N": (.4, .5, 1.5, 1.6), "G": (.3, .4, 1.6, 1.7), "W": (.25, .35, 1.65, 1.75)}


def load_input(path, *, sample_id=None):
    """Read the public aggregate format; preserve full raw covariance and seeds."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
        role = "training" if meta["group"] == "T" else "validation"
        seeds = archive["seeds"]
        actual = seeds[seeds >= 0]
        if not np.issubdtype(seeds.dtype, np.integer) or len(np.unique(actual)) != len(actual):
            raise ValueError("the W aggregate requires distinct integer physical seeds")
        metadata = dict(meta, source_file=path.name, source_sha256=file_hash(path))
        return NumericalInput(meta["labels"], archive["mean"], archive["cov"],
                              sample_id or f"w-predictivity-{meta['charge']}-{meta['group']}",
                              role=role, metadata=metadata,
                              run_ids=tuple(f"nnlojet:{i}" for i in actual))


def _smooth(t):
    t = np.clip(t, 0, 1)
    return t**3 * (t * (6 * t - 15) + 10)


def build_blocks(events, data, charge, *, definitions=None):
    """Build both declared recoil anchors and all preregistered physical blocks.

    Input events must already satisfy the baseline selection. Four-vectors fix
    the angle convention. The supplied original weights set prior-dependent
    blended targets; direct windowed targets have no such dependence.
    """
    charge = charge.lower()
    order = ORDERS[charge]
    e = coordinates(events["l_born"], events["nu"])
    # Preserve the original stored production observables where available.
    mass = np.asarray(events.get("m_w", e["m_w"]))
    rapidity = np.asarray(events.get("y_abs", e["y_abs"]))
    qt = np.asarray(events.get("pt_w", e["pt_w"]))
    c = e["c"]
    p = np.asarray(events["weight"], float)
    if not np.isfinite(p).all() or (p <= 0).any() or not np.isfinite(p.sum()):
        raise ValueError("positive finite prior weights are required")
    p = p / p.sum()
    cm = cheb(umap(mass, 40, 1000, ("bw", 80.385, 2.085)), order["m_w"])
    cy = cheb(umap(rapidity, 0, 4, "lin"), max(order["y_abs"], 4))
    cp = cheb(umap(qt, 25, 500, "log"), order["pt_w"])
    cc = cheb(c, 10)
    wq, wc = profile_w(qt, 25, 50, 500), profile_w(c + 1, .3, .4, 1.6, 1.7)
    blocks = {}

    def direct(name, values, labels, **metadata):
        blocks[name] = ConstraintBlock.from_input(name, values, data, labels=labels,
                                                  metadata=dict(charge=charge, **metadata))

    direct("mass", cm[:, 1:], tuple(f"prof_mw_{i}" for i in range(1, order["m_w"] + 1)),
           definition="production mass Chebyshev moments", map=["bw", 80.385, 2.085], range_GeV=[40, 1000])
    direct("rapidity", cy[:, 1:order["y_abs"] + 1], tuple(f"prof_absyw_{i}" for i in range(1, order["y_abs"] + 1)),
           definition="absolute production rapidity Chebyshev moments", map="linear", range=[0, 4])
    recoil_labels = tuple(f"prof_wptw_{i}" for i in range(order["pt_w"] + 1))
    direct("recoil_window", wq[:, None] * cp, recoil_labels,
           definition="direct windowed recoil moments; rate included once", transition="windowed",
           map="log", range_GeV=[25, 500], profile_GeV=[25, 50, 500])
    inside = (qt < 500).astype(float)
    values = cp[:, 1:] * inside[:, None]
    below = (1 - wq)[:, None] * values
    denominator = 1 - p @ wq
    if denominator <= 0:
        raise ValueError("the blended recoil anchor requires nonzero prior mass outside its window")
    offset = p @ below / denominator
    window_row = data.selector(("prof_wptw_0",))[0]
    matrix = data.selector(recoil_labels[1:]) - offset[:, None] * window_row
    rate = float(window_row @ data.normalized[:, 0])
    pd = (1 - rate) / denominator * (below - (1 - wq)[:, None] * offset)
    labels = tuple(f"blend_pt_T{i}" for i in range(1, order["pt_w"] + 1))
    if np.any(inside == 0):
        above = 1 - inside
        part = p @ above / denominator
        values = np.column_stack([values, above])
        offset = np.r_[offset, part]
        matrix = np.vstack([matrix, -part * window_row])
        pd = np.column_stack([pd, (1 - rate) / denominator * (above - part * (1 - wq))])
        labels += ("blend_pt_above",)
    blocks["recoil_blend"] = ConstraintBlock("recoil_blend", values, matrix, labels, offset, pd,
                                             metadata=dict(charge=charge, transition="blend", map="log",
                                                           range_GeV=[25, 500], profile_GeV=[25, 50, 500],
                                                           definition="original prior-dependent continuation; separate control for the windowed anchor"))
    direct("angle", wc[:, None] * cc[:, :7], tuple(f"prof_wctw_{i}" for i in range(7)),
           definition="original polar-angle window through degree six; rate once", profile=[.3, .4, 1.6, 1.7])
    for degree in (2, 4):
        indices = [(k, n) for k in range(1, degree + 1) for n in range(degree + 1)]
        direct(f"joint{degree}", np.column_stack([wc * cy[:, k] * cc[:, n] for k, n in indices]),
               tuple(f"prof_yc_{k}{n}" for k, n in indices),
               definition="original-window rapidity-angle products; no duplicate window rate", degrees=indices)
    definitions = definitions or json.loads((DATA / "feature_definitions.json").read_text())
    gate = _smooth(np.log(np.maximum(.5 * e["m_w"] * np.sqrt(np.maximum(1 - c * c, 0)), 1e-30) / 25) / np.log(28 / 25))
    rho, sx, sy, ax, ay = (e[k] for k in ("rho", "sx", "sy", "rho_sx", "rho_sy"))
    p2 = .5 * (1 - 3 * c * c)
    hard = wq[:, None] * np.column_stack([p2, c, 2 * c * sx, sx * sx - sy * sy, sx])
    soft = np.column_stack([rho * rho * p2, rho * rho * c, 2 * c * ax, ax * ax - ay * ay, ax])
    new_mass = cheb(umap(e["m_w"], 40, 1000, ("bw", 80.385, 2.085)), 2)
    new_y = cheb(umap(e["y_abs"], 0, 4, "lin"), 2)
    vectors, selected_rows = {}, {}
    for row in definitions:
        if row["role"] != "fit":
            continue
        spec, name = row["metadata"], row["name"]
        group = spec["block"]
        if group == "D":
            value = wc * cc[:, spec["n"]]
        elif group == "M":
            value = wc * new_mass[:, spec["i"]] * cc[:, spec["j"]]
        elif group in WINDOWS:
            value = gate * profile_w(c + 1, *WINDOWS[group]) * cc[:, spec["n"]]
            if "k" in spec:
                value *= new_y[:, spec["k"]]
        else:
            value = (hard if group == "H" else soft)[:, spec["h"]] * new_y[:, spec["k"]]
        vectors[name], selected_rows[name] = value, row
    for group in ("D", "M", "H", "S"):
        labels = tuple(name for name, row in selected_rows.items() if row["metadata"]["block"] == group)
        direct(group, np.column_stack([vectors[k] for k in labels]), labels,
               definition="declared physical-function block", measurement_functions=[selected_rows[k] for k in labels],
               required_recoil_anchor="windowed" if group in {"H", "S"} else "either")
        if group in {"H", "S"}:
            labels0 = tuple(k for k in labels if selected_rows[k]["metadata"]["k"] == 0)
            direct(group + "0", np.column_stack([vectors[k] for k in labels0]), labels0,
                   definition="unconditioned spin block", measurement_functions=[selected_rows[k] for k in labels0],
                   required_recoil_anchor="windowed")
    for window in WINDOWS:
        for kind, joint in (("angle", False), ("joint", True)):
            labels = tuple(name for name, row in selected_rows.items()
                           if row["metadata"]["block"] == window and ("k" in row["metadata"]) == joint)
            direct(f"{window}_{kind}", np.column_stack([vectors[k] for k in labels]), labels,
                   definition="mass-gated replacement angular window", measurement_functions=[selected_rows[k] for k in labels],
                   profile=WINDOWS[window], born_transverse_gate_GeV=[25, 28], window_rate_once=not joint)
    return blocks


def candidate_blocks(blocks, charge, name):
    """Return a declared candidate, enforcing compatible recoil/spin anchors.

    ``anchor`` is the original blended prescription; ``R`` replaces it by the
    direct windowed recoil prescription. D/M/window/J4 controls accept either
    anchor (``D`` or ``R+D``). H/H0/S/S0 always use R. Combined block additions
    start with R and use ``R+D+M+H+S`` syntax, subject to the study's training rule.
    """
    order = ORDERS[charge.lower()]
    parts = name.split("+")
    if name in {"H", "H0", "S", "S0"}:
        parts = ["R", name]
    windowed = parts[0] == "R"
    additions = [p for p in parts if p not in {"anchor", "R"}]
    if any(p in {"H", "H0", "S", "S0"} for p in additions) and not windowed:
        raise ValueError("spin/recoil blocks require the explicit windowed recoil anchor R")
    replacements = [p for p in additions if p in WINDOWS]
    if len(replacements) > 1 or (replacements and len(additions) != 1):
        raise ValueError("window replacements are separate declared controls")
    result = [blocks["mass"], blocks["rapidity"], blocks["recoil_window" if windowed else "recoil_blend"]]
    if replacements:
        result.append(blocks[f"{replacements[0]}_angle"])
        if order["joint"]:
            result.append(blocks[f"{replacements[0]}_joint"])
        return result
    result.append(blocks["angle"])
    if "J4" in additions:
        result.append(blocks["joint4"])
    elif order["joint"]:
        result.append(blocks[f"joint{order['joint']}"])
    for group in additions:
        if group != "J4":
            if group not in {"D", "M", "H", "H0", "S", "S0"}:
                raise ValueError(f"unknown declared candidate addition {group}")
            result.append(blocks[group])
    return result
