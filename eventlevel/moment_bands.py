#!/usr/bin/env python3
r"""Propagate moment-level uncertainties into distribution-level bands/bars.

Two error sources, kept separate all the way to the figures:
  scale -- the 7-point variation of the fixed-order targets.  Each scale set
           is a full warm-started re-solve; the band is the per-bin envelope.
  stat  -- the seed-scatter of the targets.  Propagated by bootstrap over the
           40 independent NNLOJET runs (preserves cross-moment correlations),
           each bootstrap a warm-started re-solve; the error bar is the
           per-bin standard deviation, combined in quadrature with the
           reweighted sample's own Monte Carlo error.

A linearised alternative for the scale band -- the first-order response
  dq_i = q_i (dlam . (Phz_i - <Phz>_q)),   dlam = H^{-1} dmu_z
-- is implemented for cross-checking (see __main__): it is trusted only where
it reproduces the full re-solve envelope.

Run as a script to execute the LINEARISATION CHECK on the Drell-Yan config:
full 7-point re-solves against the linear response, compared bin by bin on
m_ll and phi*.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from maxent_upgrade import upgrade
from nnlojet_moments import (fo_moments_smooth_from_nnlojet, common_seeds,
                             _moment_over_seeds, _sum_channels)

NSCALE = 7


# --------------------------------------------------------------------------
# loading: all scale sets + per-seed matrices for the bootstrap
# --------------------------------------------------------------------------
def moment_sets(base, run, channels, seeds, loader_kw):
    """[M_scale0 .. M_scale6] with identical construction, one per scale."""
    return [fo_moments_smooth_from_nnlojet(base, run, channels, seeds,
                                           scale_idx=s, **loader_kw)
            for s in range(NSCALE)]


def per_seed_table(base, run, channels, seeds, born_tags, n_born, n_recoil,
                   norm_born="norm_born", w0="prof_wptz_0", wtag="prof_wptz",
                   prefix="Z"):
    """Per-seed central-scale values of every constrained moment (+ rate),
    for bootstrap resampling.  Returns dict tag -> array [nseed]."""
    out = {}
    for cfg_obs, tag in born_tags.items():
        for n in range(1, n_born + 1):
            m = _moment_over_seeds(base, run, channels, seeds,
                                   f"prof_{tag}_{n}", norm_born, prefix)
            out[("born", cfg_obs, n)] = m[1][:, 0]
    for n in range(1, n_recoil + 1):
        m = _moment_over_seeds(base, run, channels, seeds, f"{wtag}_{n}", w0, prefix)
        out[("recoil", n)] = m[1][:, 0]
    rr = []
    for s in seeds:
        wnum = _sum_channels(base, run, channels, w0, s, prefix)
        nb = _sum_channels(base, run, channels, norm_born, s, prefix)
        rr.append(wnum[0] / nb[0] if (wnum is not None and nb is not None) else np.nan)
    out[("rate",)] = np.asarray(rr, float)
    return out


def bootstrap_moments(M0, table, rng):
    """One bootstrap replica of the central moment set M0."""
    nseed = len(next(iter(table.values())))
    idx = rng.integers(0, nseed, nseed)
    M = {"born": {}, "recoil": {}}
    for obs, d in M0["born"].items():
        vals = [float(np.nanmean(table[("born", obs, n + 1)][idx]))
                for n in range(len(d["values"]))]
        M["born"][obs] = {**d, "values": vals}
    for obs, d in M0["recoil"].items():
        vals = [float(np.nanmean(table[("recoil", n + 1)][idx]))
                for n in range(len(d["window_values"]))]
        R = float(np.nanmean(table[("rate",)][idx]))
        M["recoil"][obs] = {**d, "window_values": vals, "wprofile_values": vals,
                            "rate": R, "wprofile_rate": R}
    return M


# --------------------------------------------------------------------------
# solving: central + variants, all warm-started
# --------------------------------------------------------------------------
def band_solve(ev, Msets, cfg, table=None, n_boot=30, seed=1):
    """Central solve + 6 scale re-solves + n_boot bootstrap re-solves.
    Returns dict with 'central' (UpgradeResult), 'scale_w' [6, N], 'boot_w'."""
    res0 = upgrade(ev, Msets[0], {**cfg, "keep_features": True})
    lam0 = res0.report["lam"]
    scale_w = []
    for s in range(1, NSCALE):
        r = upgrade(ev, Msets[s], {**cfg, "lam0": lam0})
        scale_w.append(r.weights)
    boot_w = []
    if table is not None and n_boot:
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            Mb = bootstrap_moments(Msets[0], table, rng)
            r = upgrade(ev, Mb, {**cfg, "lam0": lam0})
            boot_w.append(r.weights)
    return dict(central=res0, scale_w=np.array(scale_w),
                boot_w=(np.array(boot_w) if boot_w else None))


# --------------------------------------------------------------------------
# per-bin bands from variant weights
# --------------------------------------------------------------------------
def dens(x, w, e):
    h, _ = np.histogram(x, e, weights=w / w.sum())
    return h / np.diff(e)


def band_hists(x, sol, e):
    """(central, scale_lo, scale_hi, stat_sigma, mc_sigma) densities on edges e."""
    h0 = dens(x, sol["central"].weights, e)
    hs = np.array([dens(x, w, e) for w in sol["scale_w"]])
    lo = np.minimum(h0, hs.min(0)); hi = np.maximum(h0, hs.max(0))
    stat = (np.array([dens(x, w, e) for w in sol["boot_w"]]).std(0, ddof=1)
            if sol["boot_w"] is not None else np.zeros_like(h0))
    # the reweighted sample's own MC error per bin
    q = sol["central"].weights / sol["central"].weights.sum()
    s2, _ = np.histogram(x, e, weights=q * q)
    mc = np.sqrt(s2) / np.diff(e)
    return h0, lo, hi, stat, mc


# --------------------------------------------------------------------------
# linear response (for the check)
# --------------------------------------------------------------------------
def linear_variant_weights(res0, mu_new):
    """First-order variant weights from the central solve, no re-solve."""
    Phi, p = res0.report["Phi"], res0.report["p"]
    lam, mu = res0.report["lam"], res0.report["mu"]
    q = res0.weights / res0.weights.sum()
    m_L = (p[:, None] * Phi).sum(0)
    sd = np.maximum(np.sqrt((p[:, None] * (Phi - m_L) ** 2).sum(0) + 1e-30), 1e-12)
    Phz = (Phi - m_L) / sd
    Phc = Phz - (q @ Phz)
    H = (q[:, None] * Phc).T @ Phc + 1e-4 * np.eye(Phi.shape[1])
    dmu_z = (np.asarray(mu_new, float) - mu) / sd
    dlam = np.linalg.solve(H, dmu_z)
    qn = q * (1.0 + Phc @ dlam)
    qn = np.maximum(qn, 0)
    return qn / qn.sum()


def targets_of(M, cfg_like_names):
    """Flatten a moment set into the mu vector in the solver's ordering.
    Uses the feature_names of the central solve to stay aligned."""
    raise NotImplementedError  # ordering is taken from report in the check below


# --------------------------------------------------------------------------
# the check
# --------------------------------------------------------------------------
def main():
    BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_poc"
    CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
    XM, XHI, SOFT = 30.0, 500.0, 0.5
    loader_kw = dict(born_tags={"mll": "mll", "y_abs": "absyz"},
                     n_born=6, n_recoil=12, x_match=XM, x_hi=XHI, soft_lo=SOFT)
    seeds = common_seeds(BASE, "DY_MOMENTS", CH6)
    print(f"seeds {len(seeds)}; loading 7 scale sets ...", flush=True)
    Msets = moment_sets(BASE, "DY_MOMENTS", CH6, seeds, loader_kw)

    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v3.npz")))
    n = len(P["w"]); idx = np.random.default_rng(0).choice(n, min(1_000_000, n), replace=False)
    ev = dict(mll=P["mll"][idx].astype(float), y_abs=np.abs(P["y_ll"][idx]).astype(float),
              pT_ll=P["pT_ll"][idx].astype(float), phistar=P["phistar"][idx].astype(float),
              weight=P["w"][idx].astype(float))
    cfg = dict(born={"mll": {"range": (66., 116.), "map": "bw"},
                     "y_abs": {"range": (0., 2.4), "map": "lin"}},
               recoil={"pT_ll": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                                 "profile": {"a": XM, "b": 2 * XM, "c": XHI}}},
               followers=["phistar"])

    print("central + 6 full scale re-solves ...", flush=True)
    sol = band_solve(ev, Msets, cfg, table=None, n_boot=0)
    res0 = sol["central"]
    print(f"  central: {res0.summary()}")

    # linearised variants: mu vectors of the scale sets, in solver ordering,
    # via re-running the feature assembly through upgrade's report
    lin_w = []
    for s in range(1, NSCALE):
        r = upgrade(ev, Msets[s], {**cfg, "lam0": res0.report["lam"]})
        mu_new = r.report["mu"]          # exact ordering guaranteed
        lin_w.append(linear_variant_weights(res0, mu_new))
    lin_w = np.array(lin_w)

    for key, e in (("mll", np.linspace(66, 116, 26)),
                   ("phistar", np.geomspace(4e-3, 10, 37))):
        h0 = dens(ev[key], res0.weights, e)
        full = np.array([dens(ev[key], w, e) for w in sol["scale_w"]])
        lin = np.array([dens(ev[key], w, e) for w in lin_w])
        width = np.maximum(full.max(0) - full.min(0), 1e-30)
        d_lo = np.abs(full.min(0) - lin.min(0)) / width
        d_hi = np.abs(full.max(0) - lin.max(0)) / width
        ok = h0 > 0
        print(f"  {key:8s} band width (median, rel to central) "
              f"{np.median((width[ok]) / h0[ok]) * 100:5.2f}%   "
              f"edge diff lin vs full: median {100*np.median(np.r_[d_lo[ok], d_hi[ok]]):.1f}%"
              f"  max {100*np.max(np.r_[d_lo[ok], d_hi[ok]]):.1f}% of band width")


if __name__ == "__main__":
    main()
