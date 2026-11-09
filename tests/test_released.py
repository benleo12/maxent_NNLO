#!/usr/bin/env python3
r"""Checks of the released moment tables and the loader (no event samples needed).

    python tests/test_released.py        (or: pytest tests/)

1. every table loads into (moments, config) with consistent keys, the default depths and
   transitions (Z+jet windowed, everything else blended), and carries its provenance;
2. pooling the per-run accumulators (<name>_runs.json.gz) over all runs at the central
   scale reproduces every released value, Monte Carlo error, scale error and window rate,
   for the main tables and the matching-window variants;
3. the window variants replace the recoil tower (for Z+jet both jet towers and their joint
   block) and nothing else, and are not seam-checked;
4. the Higgs cross section is the NNLO rate of the on-shell production;
5. compute_fo_moments honours the window: moments computed from a sample, imposed on the
   same sample, leave it unchanged (q = p), in either transition, also for a window that
   starts below the lower end of its map (the Drell-Yan and W variants A), where the booked
   measurement function is w(v) T_n(-1);
6. the CT18NNLO member tables shift the imposed towers by member k - member 0, and
   hessian_band forms the CT18 band;
7. the Z+jet acoplanarity tower of the paper loads with its 4 default orders, windowed.
"""
import contextlib
import io
import json
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
        obs = [t["towers"][k]["observable"] for k in rec]
        first = [k for i, k in enumerate(rec) if obs[i] not in obs[:i]]    # one tower per observable
        for r in [first] + [[k] for k in rec if k not in first]:
            kw = dict(born=born, recoil=r, mixed=list(t["joint"]), max_orders={k: 99 for k in t["towers"]},
                      check_seam=False)
            a, _ = load_moments(p, **kw, qt_basis="booked")
            b, _ = load_moments(p, scale=0, runs=list(range(n)), **kw, qt_basis="booked")
            worst = _compare(a, b, worst)
        for w in _pooled_windows(p):
            a, _ = load_moments(p, window=w, check_seam=False, qt_basis="booked")
            b, _ = load_moments(p, window=w, scale=0, runs=list(range(n)), check_seam=False, qt_basis="booked")
            worst = _compare(a, b, worst)
    assert worst < 1e-8, worst


def _pooled_windows(p):
    """the windows booked outside the main table with per-run accumulators (the others are central-scale only)"""
    out = []
    for w in released.WINDOWS.get(p, {}):
        g = released._window(p, w)
        if g and all("numerator_tags" in d for d in g[0].values()):
            out.append(w)
    return out


def test_matching_windows():
    for p, towers in (("ggh", ["pT_H"]), ("aa", ["pt_aa"]), ("dy", ["pT_ll"]), ("wp", ["pt_w"]), ("wm", ["pt_w"]),
                      ("zj", ["ptj1", "ptj2"])):
        nominal, cn = load_moments(p, check_seam=False, qt_basis="booked")
        wins = released.windows(p)
        assert list(wins) == ["nominal", "down", "up"], (p, list(wins))
        assert wins["nominal"] == cn["recoil"][towers[0]]["profile"]
        assert wins["down"]["a"] < wins["nominal"]["a"] < wins["up"]["a"]
        for w in ("down", "up"):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                m, c = load_moments(p, window=w, qt_basis="booked")                    # the seam check stays quiet
            assert out.getvalue() == "", out.getvalue()
            g = released._window(p, w)
            for tower in towers:
                prof = c["recoil"][tower]["profile"]
                assert prof == wins[w] and m["recoil"][tower]["x_match"] == prof["a"]
                d = g[0][tower] if g else released.table(p)["towers"][tower]
                assert m["recoil"][tower]["rate"] == d["window_rate"]
                assert m["recoil"][tower]["window_values"] == d["values"][:len(m["recoil"][tower]["window_values"])]
                assert m["recoil"][tower]["window_values"] != nominal["recoil"][tower]["window_values"]
            assert m["born"] == nominal["born"] and c["born"] == cn["born"]
            for key, j in (g[1] if g else {}).items():               # Z+jet: the joint block moves too
                assert m["mixed"][key]["rate"] == j["window_rate"]
                assert c["mixed"][key]["profile"] == j["profile"]
                assert {o: pr["a"] for o, pr in j["profile"].items()} == {o: wins[w]["a"] for o in towers}
            if g and w not in _pooled_windows(p):
                for kw in (dict(scale=1), dict(runs=[0, 1])):
                    try:
                        load_moments(p, window=w, **kw, qt_basis="booked")
                    except ValueError as e:
                        assert "central scale only" in str(e), str(e)
                    else:
                        raise AssertionError(f"{p} window {w} is central-scale only")
        m, c = load_moments(p, window="nominal", check_seam=False, qt_basis="booked")
        assert m == nominal and c == cn
    # the Drell-Yan and W lower windows start below the lower end of the log map (30 and 25 GeV)
    for p, tower, a in (("dy", "pT_ll", 20.0), ("wp", "pt_w", 17.0), ("wm", "pt_w", 17.0)):
        _, c = load_moments(p, window="down", qt_basis="booked")
        assert c["recoil"][tower]["profile"]["a"] == a < c["recoil"][tower]["soft_lo"]
    for bad in (lambda: load_moments("aa", window="C", qt_basis="booked"), lambda: load_moments("ggh", window="D", qt_basis="booked"),
                lambda: load_moments("ggh", recoil=[], window="up", qt_basis="booked")):
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


def test_window_below_map():
    """A window that starts below the lower end of its log map (Drell-Yan A: 20-40 GeV on [30, 2500]):
    the booked moment is <w(v) T_n(clip(u(v), -1, 1))> / <w>, and the package evaluates the prior the
    same way, so moments booked on a sample and imposed on it leave it unchanged."""
    rng = np.random.default_rng(5); n = 200_000
    ev = dict(pt=rng.exponential(25, n).clip(1, 2000), weight=rng.uniform(0.5, 1.5, n))
    lo, hi, a, b, c = 30.0, 2500.0, 20.0, 40.0, 500.0
    t = np.clip((np.log(ev["pt"]) - np.log(a)) / np.log(b / a), 0, 1)
    w = (6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3) * (ev["pt"] < c) * ev["weight"]
    u = np.clip(2 * (np.log(ev["pt"]) - np.log(lo)) / np.log(hi / lo) - 1, -1, 1)
    direct = [float(np.sum(w * np.cos(k * np.arccos(u))) / np.sum(w)) for k in range(1, 7)]
    assert (w[ev["pt"] < lo] > 0).any()
    for transition in ("blend", "windowed"):
        cfg = dict(recoil={"pt": {"range": (lo, hi), "map": "log", "soft_lo": lo,
                                  "profile": {"a": a, "b": b, "c": c}, "transition": transition}})
        mom = compute_fo_moments(ev, cfg, n_max=6)
        assert np.allclose(mom["recoil"]["pt"]["window_values"], direct, rtol=0, atol=1e-12)
        r = upgrade(ev, mom, {**cfg, "moment_selection": False})
        p = ev["weight"] / ev["weight"].sum()
        assert np.max(np.abs(r.weights / p - 1)) < 1e-6, (transition, np.max(np.abs(r.weights / p - 1)))


def test_pdf_members():
    for p, recoil in (("wp", None), ("wm", ["pt_w", "ctw_window"]), ("ggh", None)):
        pt = (json.load(open(released._path(released.PDF_MEMBERS_NOMINAL[p]))) if p in released.PDF_MEMBERS_NOMINAL
              else released.pdf_table(p))                  # Higgs: the members of the nominal 54 GeV window
        assert pt["n_members"] == 59 == len(pt["sigma_fid_ratio"]) and pt["sigma_fid_ratio"][0] == 1.0
        m0, c0 = load_moments(p, recoil=recoil, check_seam=False, qt_basis="booked")
        mm, cm = load_moments(p, recoil=recoil, pdf_member=0, qt_basis="booked")
        assert mm == m0 and cm == c0
        for k in (1, 30, 58):
            mk, ck = load_moments(p, recoil=recoil, pdf_member=k, qt_basis="booked")
            assert ck == c0
            for key, d in pt["towers"].items():
                o, dv = d["observable"], np.subtract(d["values"][k], d["values"][0])
                assert len(d["values"]) == 59 and np.any(dv != 0), (p, key)
                for role, vk in (("born", "values"), ("recoil", "window_values")):
                    if o in m0[role]:
                        x0, xk = np.asarray(m0[role][o][vk]), np.asarray(mk[role][o][vk])
                        assert np.allclose(xk - x0, dv[:len(x0)], rtol=0, atol=1e-15), (p, key, k)
                        if role == "recoil":
                            assert mk[role][o]["rate"] == m0[role][o]["rate"] + (d["window_rate"][k] - d["window_rate"][0])
        if recoil is None and p != "ggh":                         # a tower not imposed is not shifted
            assert "ctw1" not in load_moments(p, pdf_member=3, qt_basis="booked")[0]["recoil"]
    for bad in (lambda: load_moments("dy", pdf_member=1, qt_basis="booked"), lambda: load_moments("wp", pdf_member=59, qt_basis="booked"),
                lambda: load_moments("wp", pdf_member=1, scale=2, qt_basis="booked"), lambda: load_moments("ggh", pdf_member=1, window="up", qt_basis="booked")):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("an invalid pdf_member request must raise")
    # the CT18 Hessian band: pairs (2i-1, 2i) about member 0, 90% CL / 1.645 for 68% CL
    x = np.zeros((59, 2)); x[1] = [1.0, -2.0]; x[2] = [-3.0, 0.5]; x[57] = [0.0, 4.0]
    up, dn = released.hessian_band(x, cl68=False)
    assert np.allclose(up, [1.0, np.hypot(0.5, 4.0)]) and np.allclose(dn, [3.0, 2.0])
    assert np.allclose(released.hessian_band(x)[0], up / 1.645)


def test_acoplanarity_tower():
    t = released.table("zj")["towers"]["aco"]
    assert t["observable"] == "pimdphi" and t["profile"]["a"] > 2 * np.arctan(30 / 91.1876)   # above v_m = 30 GeV
    assert abs(t["profile"]["a"] - 0.68722) < 1e-5 and t["profile"]["b"] == 2 * t["profile"]["a"]
    assert t["range"][0] == t["soft_lo"] == t["profile"]["a"] and t["range"][1] == np.pi
    m, c = load_moments("zj", recoil=["ptj1", "ptj2", "aco"])
    assert len(m["recoil"]["pimdphi"]["window_values"]) == 4 == len(t["values"][:4])
    assert all(r["transition"] == "windowed" for r in c["recoil"].values())
    assert c["recoil"]["pimdphi"]["profile"] == t["profile"] and "mixed" in c
    assert m["recoil"]["pimdphi"]["rate"] == t["window_rate"]
    m, c = load_moments("zj", recoil=["ptj1", "ptj2", "aco"], window="up")     # with the jet windows moved up
    assert c["recoil"]["pimdphi"]["profile"] == t["profile"] and c["recoil"]["ptj1"]["profile"]["a"] == 40.0
    try:
        load_moments("zj", recoil=["ptj1", "ptj2", "pimdphi", "aco"])
    except ValueError as e:
        assert "impose one of them" in str(e)
    else:
        raise AssertionError("two towers of one observable must raise")


def test_higgs_window_from_color_factor():
    """The nominal Higgs window starts at the color-corrected matching scale (C_A/C_F = 9/4), within 1 GeV,
    and its lower window (the main table's 37 GeV one) at the color-blind one."""
    from maxent_upgrade import matching_scale
    mH = float(released.table("ggh")["hard_scale_GeV"])
    assert abs(released.windows("ggh")["nominal"]["a"] - matching_scale(mH, color=9 / 4)) < 1.0
    assert abs(released.windows("ggh")["down"]["a"] - matching_scale(mH)) < 1.0


if __name__ == "__main__":
    for f in (test_tables_load, test_higgs_cross_section, test_runs_reproduce_tables, test_matching_windows,
              test_window_consistency, test_window_below_map, test_pdf_members, test_acoplanarity_tower,
              test_higgs_window_from_color_factor):
        f(); print(f"{f.__name__}: PASS")
