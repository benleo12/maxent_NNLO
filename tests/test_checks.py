#!/usr/bin/env python3
r"""Input checks and solver safeguards, on toy samples.

    python tests/test_checks.py        (or: pytest tests/)

1. a Breit-Wigner map given as ('bw', M) takes the default width, as 'bw' does;
2. a recoil tower without 'x_hi' is mapped on the interval the moments were booked on;
3. load_moments refuses an unknown process, scale or run list with a clear message, and
   swaps in the NNLO Z+jet recoil tower for Drell-Yan only;
4. the solver gives the same weights for a prior that does not sum to one;
5. a warm start of the wrong length is refused with a clear message.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maxent_upgrade import upgrade, compute_fo_moments, load_moments   # noqa: E402
from maxent_upgrade.basis import cheb, umap                             # noqa: E402
from maxent_upgrade.maxent_match import _newton_maxent                  # noqa: E402

LO, HI = 1.0, 1000.0


def _raises(fn, match):
    try:
        fn()
    except ValueError as e:
        assert match in str(e), str(e)
        return
    raise AssertionError(f"no ValueError mentioning {match!r}")


def _sample(n=200_000, seed=7):
    rng = np.random.default_rng(seed)
    pt = np.exp(rng.normal(np.log(12.0), 1.1, n)).clip(LO, 1500.0)
    return dict(pt=pt, weight=rng.uniform(0.5, 1.5, n))


def _fo(ev, slope=0.3):
    w = ev["weight"] * np.exp(slope * np.log(np.maximum(ev["pt"], 5.0) / 30.0))
    return dict(ev, weight=w)


def test_bw_two_tuple_takes_default_width():
    x = np.linspace(60.0, 120.0, 101)
    assert np.array_equal(umap(x, 66.0, 116.0, ("bw", 91.1876)), umap(x, 66.0, 116.0, "bw"))


def test_recoil_without_x_hi_uses_booked_interval():
    ev = _sample()
    cfg = dict(recoil={"pt": {"range": (LO, HI), "map": "log", "soft_lo": LO,
                              "profile": {"a": 20.0, "b": 40.0, "c": 500.0}}},
               moment_selection=False)
    mom = compute_fo_moments(_fo(ev), cfg, n_max=6)
    with_hi = upgrade(ev, mom, cfg)
    mom_no_hi = {**mom, "recoil": {"pt": {k: v for k, v in mom["recoil"]["pt"].items() if k != "x_hi"}}}
    without_hi = upgrade(ev, mom_no_hi, cfg)
    # the same problem: both solves stop at the solver's round-off floor
    assert np.max(np.abs(without_hi.weights / with_hi.weights - 1)) < 1e-7
    assert without_hi.report["windows"]["pt"]["x_hi"] == HI


def test_load_moments_rejects_bad_arguments():
    _raises(lambda: load_moments("DY"), "released processes")
    _raises(lambda: load_moments("dy", scale=7), "scale")
    _raises(lambda: load_moments("dy", runs=[]), "runs")
    _raises(lambda: load_moments("dy", runs=[-1, 0]), "runs")


def test_nnlo_zj_recoil():
    moments, config = load_moments("dy", nnlo_zj=True)
    rc = moments["recoil"]["pT_ll"]
    assert len(rc["window_values"]) == 9
    assert config["recoil"]["pT_ll"]["profile"] == {"a": 30.0, "b": 60.0, "c": 500.0}
    assert 0.09 < rc["rate"] < 0.12
    assert load_moments("dy", nnlo_zj=True, scale=3)[0]["recoil"]["pT_ll"]["rate"] != rc["rate"]
    _raises(lambda: load_moments("wp", nnlo_zj=True), "Drell-Yan")


def _toy_problem(n=50_000, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    Phi = cheb(x, 4)[:, 1:]
    target = (1.0 + 0.3 * x) / 2.0                  # a smooth density on [-1, 1]
    mu = (Phi * target[:, None]).mean(0) * 2.0      # its first four Chebyshev moments
    return Phi, np.full(n, 1.0 / n), mu


def test_unnormalised_prior_gives_same_weights():
    Phi, p, mu = _toy_problem()
    q1, lam1, ok1 = _newton_maxent(Phi, p, mu)
    q2, lam2, ok2 = _newton_maxent(Phi, 3.7 * p, mu)
    assert ok1 and ok2
    assert np.max(np.abs(q2 / q1 - 1)) < 1e-12


def test_warm_start_length_is_checked():
    Phi, p, mu = _toy_problem()
    _raises(lambda: _newton_maxent(Phi, p, mu, lam0=np.zeros(Phi.shape[1] + 1)), "lam0")


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print(f"{name}: PASS")
