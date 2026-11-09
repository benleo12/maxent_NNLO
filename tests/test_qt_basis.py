"""The q_T basis of the acting range (load_moments(..., qt_basis="range"), the default).

The log-mapped window towers act only between the window's lower edge and its cut.  The range
basis re-expresses their booked Chebyshev moments as Chebyshev polynomials of the log map on that
interval, an exact linear change of basis.  These tests check the acting ranges, that the new values
are the exact combination of the booked ones and agree with the per-run accumulators within a small
fraction of their errors, and that both bases give the same posterior where both are well conditioned.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from maxent_upgrade import upgrade, load_moments, compute_fo_moments, released   # noqa: E402

EXPECT = {"dy": {"pT_ll": (30.0, 500.0)}, "wp": {"pt_w": (25.0, 500.0)}, "wm": {"pt_w": (25.0, 500.0)},
          "ggh": {"pT_H": (54.0, 1000.0)}, "aa": {"pt_aa": (28.0, 500.0)},
          "zj": {"ptj1": (30.0, 1000.0), "ptj2": (30.0, 1000.0)}}


def test_acting_ranges():
    for p, exp in EXPECT.items():
        m, c = load_moments(p, check_seam=False)
        mb, cb = load_moments(p, check_seam=False, qt_basis="booked")
        for o, (lo, hi) in exp.items():
            assert c["recoil"][o]["range"] == (lo, hi) and c["recoil"][o]["soft_lo"] == lo, (p, o)
            assert m["recoil"][o]["soft_lo"] == lo and m["recoil"][o]["x_hi"] == hi, (p, o)
            assert len(m["recoil"][o]["window_values"]) == len(mb["recoil"][o]["window_values"])
            assert m["recoil"][o]["rate"] == mb["recoil"][o]["rate"]
        assert m["born"] == mb["born"] and c["born"] == cb["born"]
        if "mixed" in m:                                    # joint blocks stay as booked
            assert m["mixed"] == mb["mixed"] and c["mixed"] == cb["mixed"]
    # the letter's NNLO Z+jet tower stays as booked
    m, c = load_moments("dy", nnlo_zj=True)
    assert c["recoil"]["pT_ll"]["range"] == (30.0, 2500.0)
    # a tower already booked on its acting range is untouched, also when another tower books the same observable
    t = released.table("zj")["towers"]["aco"]
    m, c = load_moments("zj", recoil=["ptj1", "ptj2", "aco"])
    mb, _ = load_moments("zj", recoil=["ptj1", "ptj2", "aco"], qt_basis="booked")
    assert c["recoil"]["pimdphi"]["range"] == (float(t["soft_lo"]), float(t["range"][1]))
    assert m["recoil"]["pimdphi"] == mb["recoil"]["pimdphi"]


def test_exact_combination():
    worst = 0.0
    for p, exp in EXPECT.items():
        t = released.table(p)
        mb, _ = load_moments(p, check_seam=False, qt_basis="booked")
        m, _ = load_moments(p, check_seam=False)
        for o in exp:
            g = released._window(p, "nominal") if p in released.WINDOWS else None
            if g:                                       # the nominal tower lives in a window table (Higgs: 54 GeV)
                d, src = g[0][o] if o in g[0] else next(iter(g[0].values())), g[3]
            else:
                d, src = next(x for x in t["towers"].values() if x.get("observable") == o and x["kind"] == "window"), p
            lo, hi = float(d["soft_lo"]), float(d["range"][1]); L, U = released.acting_range(d)
            xm, xc = released._log_u(L, lo, hi), released._log_u(U, lo, hi)
            al = 2 / (xc - xm); N = len(mb["recoil"][o]["window_values"])
            A = released.cheb_affine(N, al, -1 - al * xm)
            # the matrix is the Chebyshev expansion of T_n(al x + be)
            x = np.linspace(-1, 1, 7)
            for n in range(N + 1):
                assert np.allclose(np.polynomial.chebyshev.chebval(x, A[n]),
                                   np.polynomial.chebyshev.chebval(al * x - 1 - al * xm, [0] * n + [1]),
                                   rtol=1e-9, atol=1e-9 * np.abs(A[n]).max())
            v = np.asarray(m["recoil"][o]["window_values"]); err = np.asarray(m["recoil"][o]["stat_errors"])
            acc, st, sc = released._range_accumulators(src, d, N, A, 0, None)
            assert np.allclose(st, err) and np.allclose(sc, m["recoil"][o]["scale_errors"])
            worst = max(worst, float(np.max(np.abs(v - acc) / err)))
    assert worst < 0.01, worst                       # rounding far below the Monte Carlo errors


def test_same_posterior():
    """A toy prior and target booked on a wide log map, imposed in both bases: the same weights."""
    rng = np.random.default_rng(11); n = 300_000
    ev = dict(pt=rng.exponential(30, n).clip(1, 3000), weight=rng.uniform(0.5, 1.5, n))
    fo = dict(pt=ev["pt"], weight=ev["weight"] * (ev["pt"] / 40.0) ** 0.15)
    lo, hi, a, b, c = 30.0, 2500.0, 30.0, 60.0, 500.0
    for transition in ("blend", "windowed"):
        cfg = dict(recoil={"pt": {"range": (lo, hi), "map": "log", "soft_lo": lo,
                                  "profile": {"a": a, "b": b, "c": c}, "transition": transition}})
        mom = compute_fo_moments(fo, cfg, n_max=6)
        mom["recoil"]["pt"]["x_hi"] = hi
        r0 = upgrade(ev, mom, {**cfg, "moment_selection": False, "L2": 0.0})
        rc = dict(mom["recoil"]["pt"])
        tower = dict(soft_lo=lo, range=(lo, hi), profile=cfg["recoil"]["pt"]["profile"])
        L, U = released.acting_range(tower)
        xm, xc = released._log_u(L, lo, hi), released._log_u(U, lo, hi); al = 2 / (xc - xm)
        A = released.cheb_affine(6, al, -1 - al * xm)
        v = A[1:, 0] + A[1:, 1:] @ np.asarray(rc["window_values"])
        rc.update(window_values=list(v), wprofile_values=list(v), soft_lo=L, x_hi=U)
        cfg1 = dict(recoil={"pt": {**cfg["recoil"]["pt"], "range": (L, U), "soft_lo": L}})
        r1 = upgrade(ev, {"recoil": {"pt": rc}}, {**cfg1, "moment_selection": False, "L2": 0.0})
        dq = np.max(np.abs(r1.weights / r0.weights - 1))
        assert dq < 1e-6, (transition, dq)


if __name__ == "__main__":
    for f in (test_acting_ranges, test_exact_combination, test_same_posterior):
        f()
        print(f"{f.__name__}: PASS")
