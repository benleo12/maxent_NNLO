#!/usr/bin/env python3
r"""Checks of the blended transition (the default for a recoil tower), on toy samples.

    python tests/test_blend.py        (or: pytest tests/)

1. the blended targets close: the q-weighted moments of the bare T_n(u(v)), clipped at the
   window's lower edge, equal c <(1 - w) T_n>_p + r <T_n>_{FO,w}, computed here independently;
2. below the window the tilt is flat: q/p takes one value for every event under v_m;
3. fixed order equal to the prior returns the prior, q = p;
4. a hard upper cut c fixes the share above c with one extra constraint and keeps q/p flat
   there; a smooth upper edge c -> d needs no extra constraint and keeps q/p flat above d;
5. a joint block that couples a blended tower is refused (joint moments are windowed only).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maxent_upgrade import upgrade, compute_fo_moments, profile_w   # noqa: E402
from maxent_upgrade.basis import cheb, umap                          # noqa: E402

A, B, LO, HI = 20.0, 40.0, 1.0, 1000.0


def _sample(n=300_000, seed=4):
    rng = np.random.default_rng(seed)
    pt = np.exp(rng.normal(np.log(12.0), 1.1, n)).clip(LO, 1500.0)
    return dict(pt=pt, weight=rng.uniform(0.5, 1.5, n))


def _fo(ev, slope=0.35):
    """A 'fixed order': the prior with a smooth hardening of the tail (and scale copies)."""
    lift = np.exp(slope * np.log(np.maximum(ev["pt"], 5.0) / 30.0))
    w = ev["weight"] * lift
    return dict(ev, weight=w, weight_scales=np.array([w * f for f in (1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98)]))


def _cfg(profile, transition="blend"):
    return dict(recoil={"pt": {"range": (LO, HI), "map": "log", "soft_lo": LO, "profile": profile,
                               "transition": transition}},
                moment_selection=False)


def test_blended_targets_close():
    ev = _sample(); prof = {"a": A, "b": B, "c": 500.0}
    cfg = _cfg(prof)
    mom = compute_fo_moments(_fo(ev), cfg, n_max=8)
    # a ridge far below the default (1e-4, a numerical stabilizer) so that the closure is exact
    r = upgrade(ev, mom, {**cfg, "L2": 1e-10})
    assert r.report["windows"]["pt"]["transition"] == "blend"
    # the blended target, computed here from its definition
    x = ev["pt"]; p = ev["weight"] / ev["weight"].sum()
    w = profile_w(x, A, B, 500.0)
    inside = (x < 500.0).astype(float)
    C = cheb(umap(np.clip(x, A, HI), LO, HI, "log"), 8)
    R = mom["recoil"]["pt"]["rate"]; Rp = float((p * w).sum()); c = (1 - R) / (1 - Rp)
    targets = [c * float((p * (1 - w) * inside * C[:, n]).sum()) + R * mom["recoil"]["pt"]["window_values"][n - 1]
               for n in range(1, 9)] + [c * float((p * (1 - inside)).sum())]
    achieved = [float((r.weights * inside * C[:, n]).sum()) for n in range(1, 9)] + [float((r.weights * (1 - inside)).sum())]
    names = r.report["feature_names"]
    solver_targets = [r.report["mu"][names.index(f"pt_T{n}")] for n in range(1, 9)] + [r.report["mu"][names.index("pt_above")]]
    assert np.max(np.abs(np.subtract(targets, solver_targets))) < 1e-12
    assert np.max(np.abs(np.subtract(achieved, targets))) < 1e-7, np.max(np.abs(np.subtract(achieved, targets)))
    # the default ridge closes to its own precision
    assert upgrade(ev, mom, cfg).closure < 1e-3


def test_flat_below_matching_scale():
    ev = _sample(); cfg = _cfg({"a": A, "b": B, "c": 500.0})
    r = upgrade(ev, compute_fo_moments(_fo(ev), cfg, n_max=8), cfg)
    t = r.weights / (ev["weight"] / ev["weight"].sum())
    below = ev["pt"] < A
    assert below.sum() > 1000
    spread = float(t[below].std() / t[below].mean())
    assert spread < 1e-10, spread
    # and the tilt is continuous at the lower edge
    x = ev["pt"]
    just_below = t[(x > 0.99 * A) & (x < A)].mean(); just_above = t[(x > A) & (x < 1.01 * A)].mean()
    assert abs(just_above / just_below - 1) < 3e-3, (just_below, just_above)


def test_fixed_order_equal_to_prior_gives_prior():
    ev = _sample()
    for prof in ({"a": A, "b": B, "c": 500.0}, {"a": A, "b": B, "c": 300.0, "d": 600.0}):
        cfg = _cfg(prof)
        r = upgrade(ev, compute_fo_moments(dict(ev), cfg, n_max=8), cfg)
        p = ev["weight"] / ev["weight"].sum()
        dev = float(np.max(np.abs(r.weights / p - 1)))
        assert dev < 1e-8, (prof, dev)


def test_upper_edges():
    ev = _sample()
    p = ev["weight"] / ev["weight"].sum()
    # hard cut at c: one extra constraint fixes the share above c, and q/p is flat there
    cfg = _cfg({"a": A, "b": B, "c": 300.0})
    r = upgrade(ev, compute_fo_moments(_fo(ev, 0.5), cfg, n_max=8), cfg)
    assert "pt_above" in r.report["feature_names"]
    t = r.weights / p; above = ev["pt"] >= 300.0
    assert above.sum() > 100
    assert float(t[above].std() / t[above].mean()) < 1e-10
    # smooth fall c -> d: no extra constraint; features clipped at d, so q/p is flat above d
    cfg = _cfg({"a": A, "b": B, "c": 300.0, "d": 600.0})
    r = upgrade(ev, compute_fo_moments(_fo(ev, 0.5), cfg, n_max=8), cfg)
    assert not any(nm.endswith("_above") for nm in r.report["feature_names"])
    t = r.weights / p; above = ev["pt"] >= 600.0
    assert above.sum() > 10
    assert float(t[above].std() / t[above].mean()) < 1e-10
    # absolute closure (the first target is close to zero, so the relative closure says little)
    r = upgrade(ev, compute_fo_moments(_fo(ev, 0.5), cfg, n_max=8), {**cfg, "keep_features": True})
    ach = r.weights @ r.report["Phi"]
    assert np.max(np.abs(ach - r.report["mu"])) < 1e-4, np.max(np.abs(ach - r.report["mu"]))


def test_joint_block_refuses_blended_tower():
    rng = np.random.default_rng(1); n = 50_000
    x = np.exp(rng.normal(np.log(40.0), 0.8, n)).clip(10, 1000); y = (x * np.exp(rng.normal(-0.7, 0.5, n))).clip(10, 1000)
    ev = dict(x=x, y=y, weight=np.ones(n))
    prof = dict(a=30.0, b=60.0, c=1000.0)
    cfg = dict(recoil={o: {"range": (10., 1000.), "map": "log", "soft_lo": 10., "profile": prof} for o in ("x", "y")},
               mixed={"xy": dict(observables=("x", "y"), range={"x": (10., 1000.), "y": (10., 1000.)},
                                 map={"x": "log", "y": "log"}, profile={"x": prof, "y": prof}, n=2)})
    mom = compute_fo_moments(ev, cfg, n_max=4)
    try:
        upgrade(ev, mom, cfg)
    except ValueError as e:
        assert "windowed" in str(e)
    else:
        raise AssertionError("a joint block coupling blended towers must raise")
    for o in ("x", "y"):
        cfg["recoil"][o]["transition"] = "windowed"
    r = upgrade(ev, mom, cfg)                      # windowed: accepted, and q = p (FO = prior)
    assert np.max(np.abs(r.weights * n - 1)) < 1e-8
    try:
        upgrade(ev, mom, {**cfg, "recoil": {"x": {**cfg["recoil"]["x"], "transition": "smooth"}, "y": cfg["recoil"]["y"]}})
    except ValueError as e:
        assert "'blend' or 'windowed'" in str(e)
    else:
        raise AssertionError("an unknown transition must raise")


if __name__ == "__main__":
    for f in (test_blended_targets_close, test_flat_below_matching_scale, test_fixed_order_equal_to_prior_gives_prior,
              test_upper_edges, test_joint_block_refuses_blended_tower):
        f(); print(f"{f.__name__}: PASS")
