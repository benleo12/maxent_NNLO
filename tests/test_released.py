#!/usr/bin/env python3
r"""Checks of the released moment tables and the loader (no event samples needed).

    python tests/test_released.py        (or: pytest tests/)

1. every table loads into (moments, config) with consistent keys, the default depths and
   transitions (Z+jet windowed, everything else blended), and carries its provenance;
2. pooling the per-run accumulators (<name>_runs.json.gz) over all runs at the central
   scale reproduces every released value, Monte Carlo error, scale error and window rate,
   for the main tables and the matching-window variants;
3. the window variants replace the recoil tower and nothing else, and are not seam-checked;
4. the Higgs cross section is the NNLO rate of the on-shell production;
5. compute_fo_moments honours the window: moments computed from a sample, imposed on the
   same sample, leave it unchanged (q = p), in either transition.
"""
import contextlib
import io
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
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
        for blk in (m.get("mixed") or {}).values():
            assert set(blk["values"]) == set(blk["errors"]) == set(blk["stat_errors"]) == set(blk["scale_errors"])
        pv = t["provenance"]
        assert pv["n_runs"] == len(pv["seeds"]) == t["n_independent_runs"], p
        assert os.path.isfile(os.path.join(ROOT, pv["runcard"])), (p, pv["runcard"])
        assert released.summary(p)
    m, c = load_moments("zj", check_seam=False)
    assert len(m["recoil"]["ptj2"]["window_values"]) == 6          # the default depth
    # Z+jet: every tower windowed; elsewhere the blend (the default)
    assert {o: r.get("transition") for o, r in c["recoil"].items()} == {"ptj1": "windowed", "ptj2": "windowed"}
    m, c = load_moments("zj", recoil=["ptj1", "ptj2", "pimdphi"], check_seam=False)
    assert all(r["transition"] == "windowed" for r in c["recoil"].values())
    assert len(m["recoil"]["pimdphi"]["window_values"]) == 4
    _, c = load_moments("zj", mixed=[], check_seam=False)          # the marginal solve
    assert all(r["transition"] == "windowed" for r in c["recoil"].values())
    for p in ("dy", "wp", "wm", "aa", "ggh"):
        _, c = load_moments(p, check_seam=False)
        assert all("transition" not in r for r in c["recoil"].values()), p
    # a joint block with a window switches the towers to the windowed form
    _, c = load_moments("wp", mixed=["pt_w_x_ctw"], check_seam=False)
    assert c["recoil"]["pt_w"]["transition"] == "windowed"
    _, c = load_moments("dy", mixed=["mll_x_abs_costh"], check_seam=False)    # no window: the blend stays
    assert "transition" not in c["recoil"]["pT_ll"]


def test_higgs_cross_section():
    s = released.sigma_fid("ggh")
    assert abs(s["value"] - 39.47) < 0.01, s
    assert "48.58" in s["note"]
    t = released.table("ggh")
    assert t["provenance"]["run_directory"] == "ggh_moments3" and t["provenance"]["decay_type"] == 3


def _compare(a, b, worst):
    for role in ("born", "recoil"):
        for o in a[role]:
            for k in ("values", "window_values", "stat_errors", "scale_errors", "rate", "rate_stat", "rate_scale"):
                if k in a[role][o]:
                    x, y = np.atleast_1d(a[role][o][k]), np.atleast_1d(b[role][o][k])
                    worst = max(worst, float(np.max(np.abs(x - y) / np.maximum(np.abs(x), 1e-12))))
    for k in a.get("mixed", {}):
        for kk, v in a["mixed"][k]["values"].items():
            worst = max(worst, abs(v - b["mixed"][k]["values"][kk]) / max(abs(v), 1e-12))
    return worst


def test_runs_reproduce_tables():
    worst = 0.0
    for p in released.processes():
        t = released.table(p); n = t["n_independent_runs"]
        born = [k for k, d in t["towers"].items() if d["kind"] != "window"]
        rec = [k for k, d in t["towers"].items() if d["kind"] == "window"]
        kw = dict(born=born, recoil=rec, mixed=list(t["joint"]), max_orders={k: 99 for k in t["towers"]}, check_seam=False)
        a, _ = load_moments(p, **kw)
        b, _ = load_moments(p, scale=0, runs=list(range(n)), **kw)
        worst = _compare(a, b, worst)
        for V in released.windows(p):
            a, _ = load_moments(p, window=V, check_seam=False)
            b, _ = load_moments(p, window=V, scale=0, runs=list(range(n)), check_seam=False)
            worst = _compare(a, b, worst)
    assert worst < 1e-8, worst


def test_window_variants():
    for p, tower in (("ggh", "pT_H"), ("aa", "pt_aa")):
        nominal, cn = load_moments(p, check_seam=False)
        wins = released.windows(p)
        assert sorted(wins) == ["A", "B", "C", "nominal"]
        wt = released.window_table(p)
        for V in ("A", "B", "C"):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                m, c = load_moments(p, window=V)                    # the seam check stays quiet
            assert out.getvalue() == "", out.getvalue()
            prof = c["recoil"][tower]["profile"]
            assert prof == wins[V] and m["recoil"][tower]["x_match"] == prof["a"]
            assert m["recoil"][tower]["rate"] == wt["windows"][V]["window_rate"]
            assert m["born"] == nominal["born"] and c["born"] == cn["born"]
            assert m["recoil"][tower]["window_values"] != nominal["recoil"][tower]["window_values"]
        m, c = load_moments(p, window="nominal", check_seam=False)
        assert m == nominal and c == cn
    for bad in (lambda: load_moments("dy", window="A"), lambda: load_moments("ggh", window="D"),
                lambda: load_moments("ggh", recoil=[], window="A")):
        try:
            bad()
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError("an invalid window request must raise")


def test_window_consistency():
    rng = np.random.default_rng(3); n = 200_000
    ev = dict(m=rng.normal(91, 3, n).clip(66, 116), pt=rng.exponential(12, n).clip(0.5, 400),
              weight=rng.uniform(0.5, 1.5, n))
    for transition in ("blend", "windowed"):
        cfg = dict(born={"m": {"range": (66., 116.), "map": "bw"}},
                   recoil={"pt": {"range": (20., 400.), "map": "log", "soft_lo": 20.,
                                  "profile": {"a": 20., "b": 40., "c": 300.}, "transition": transition}})
        mom = compute_fo_moments(ev, cfg, n_max=6)
        r = upgrade(ev, mom, {**cfg, "moment_selection": False})
        p = ev["weight"] / ev["weight"].sum()
        assert np.max(np.abs(r.weights / p - 1)) < 1e-6, (transition, np.max(np.abs(r.weights / p - 1)))
        assert r.effN_retained > 0.999999


if __name__ == "__main__":
    for f in (test_tables_load, test_higgs_cross_section, test_runs_reproduce_tables, test_window_variants,
              test_window_consistency):
        f(); print(f"{f.__name__}: PASS")
