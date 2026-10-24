#!/usr/bin/env python3
r"""Checks of the released moment tables and the loader (no event samples needed).

    python tests/test_released.py        (or: pytest tests/)

1. every table loads into (moments, config) with consistent keys and the paper's depths;
2. pooling the per-run accumulators (<name>_runs.json.gz) over all runs at the central
   scale reproduces every released value, Monte Carlo error, scale error and window rate;
3. compute_fo_moments honours the window: moments computed from a sample, imposed on the
   same sample, leave it unchanged (q = p).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maxent_upgrade import upgrade, load_moments, compute_fo_moments, released   # noqa: E402


def test_tables_load():
    for p in released.processes():
        t = released.table(p)
        m, c = load_moments(p, check_seam=False)
        for role in ("born", "recoil"):
            assert set(m[role]) == set(c[role]), (p, role)
        depth = t["default_constraints"].get("max_orders", {})
        for key in t["default_constraints"]["born"] + t["default_constraints"]["recoil"]:
            d = t["towers"][key]
            role = "born" if key in t["default_constraints"]["born"] else "recoil"
            vals = m[role][d["observable"]]["values" if role == "born" else "window_values"]
            assert len(vals) == min(depth.get(key, d["n_orders"]), d["n_orders"]), (p, key)
        assert released.summary(p)
    m, c = load_moments("zj", check_seam=False)
    assert len(m["recoil"]["ptj2"]["window_values"]) == 6          # the paper's depth
    assert released.sigma_fid("ggh") is None                      # normalization not validated


def test_runs_reproduce_tables():
    worst = 0.0
    for p in released.processes():
        t = released.table(p); n = t["n_independent_runs"]
        born = [k for k, d in t["towers"].items() if d["kind"] != "window"]
        rec = [k for k, d in t["towers"].items() if d["kind"] == "window"]
        kw = dict(born=born, recoil=rec, mixed=list(t["joint"]), max_orders={k: 99 for k in t["towers"]}, check_seam=False)
        a, _ = load_moments(p, **kw)
        b, _ = load_moments(p, scale=0, runs=list(range(n)), **kw)
        for role in ("born", "recoil"):
            for o in a[role]:
                for k in ("values", "window_values", "stat_errors", "scale_errors", "rate", "rate_stat", "rate_scale"):
                    if k in a[role][o]:
                        x, y = np.atleast_1d(a[role][o][k]), np.atleast_1d(b[role][o][k])
                        worst = max(worst, float(np.max(np.abs(x - y) / np.maximum(np.abs(x), 1e-12))))
        for k in a.get("mixed", {}):
            for kk, v in a["mixed"][k]["values"].items():
                worst = max(worst, abs(v - b["mixed"][k]["values"][kk]) / max(abs(v), 1e-12))
    assert worst < 1e-8, worst


def test_window_consistency():
    rng = np.random.default_rng(3); n = 200_000
    ev = dict(m=rng.normal(91, 3, n).clip(66, 116), pt=rng.exponential(12, n).clip(0.5, 400),
              weight=rng.uniform(0.5, 1.5, n))
    cfg = dict(born={"m": {"range": (66., 116.), "map": "bw"}},
               recoil={"pt": {"range": (20., 400.), "map": "log", "soft_lo": 20.,
                              "profile": {"a": 20., "b": 40., "c": 300.}}})
    mom = compute_fo_moments(ev, cfg, n_max=6)
    r = upgrade(ev, mom, {**cfg, "moment_selection": False})
    p = ev["weight"] / ev["weight"].sum()
    assert np.max(np.abs(r.weights / p - 1)) < 1e-6, np.max(np.abs(r.weights / p - 1))
    assert r.effN_retained > 0.999999


if __name__ == "__main__":
    for f in (test_tables_load, test_runs_reproduce_tables, test_window_consistency):
        f(); print(f"{f.__name__}: PASS")
