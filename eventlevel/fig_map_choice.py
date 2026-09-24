#!/usr/bin/env python3
r"""JHEP figure: the map is a scheme choice that matters at fixed order count.

Six Chebyshev moments of m_ll imposed on the LINEAR map versus the
BREIT-WIGNER map, everything else identical.  On the linear map the Z peak
occupies a sliver of u-space and six moments cannot describe it; on the BW map
a pure resonance is flat in u and the same six moments close.

The BW targets are the event-level NNLOJET moments (the production inputs).
The linear-map targets are computed from the fine NNLOJET m_ll histogram by
overlap integration -- NNLOJET compiles one map per production run, so the
lin-map tower is not booked event-level; the same histogram-derived estimate
reproduces the event-level BW moments at the 1e-3 level, within twice the FO
error (printed as a check), so the comparison is apples to apples.
"""
import copy
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import use_pub_style, C, LW, LS, rebin_density
use_pub_style(base=17)
from maxent_upgrade import upgrade
from nnlojet_moments import fo_moments_smooth_from_nnlojet, common_seeds, fo_curve, fo_curve_band
from bandviz import mc_err, stagger

BASE = os.path.join(os.environ.get("NNLOJET_ROOT",
        os.path.expanduser("~/nnlojet-v1.0.2")), "dy_profile_log30_hi")
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
XM, XHI, SOFT, XMAP = 30.0, 500.0, 30.0, 2500.0
MZ, GZ = 91.1876, 2.4952
A, B = 66.0, 116.0


def cheb_tower(u, nmax):
    T = [np.ones_like(u), u]
    for _ in range(2, nmax + 1):
        T.append(2 * u * T[-1] - T[-2])
    return np.array(T[1:nmax + 1])


def u_lin(x):
    return 2 * (x - A) / (B - A) - 1


def u_bw(x):
    t = np.arctan((np.asarray(x, float) ** 2 - MZ ** 2) / (MZ * GZ))
    ta, tb = np.arctan((A ** 2 - MZ ** 2) / (MZ * GZ)), np.arctan((B ** 2 - MZ ** 2) / (MZ * GZ))
    return 2 * (t - ta) / (tb - ta) - 1


def hist_moments(lo, hi, dens, umap, nmax=6):
    mid = 0.5 * (lo + hi); wgt = dens * (hi - lo)
    T = cheb_tower(umap(mid), nmax)
    return (T * wgt).sum(1) / wgt.sum()


def main():
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v3.npz")))
    n = len(P["w"]); idx = np.random.default_rng(0).choice(n, min(1_000_000, n), replace=False)
    ev = dict(mll=P["mll"][idx].astype(float), y_abs=np.abs(P["y_ll"][idx]).astype(float),
              pT_ll=P["pT_ll"][idx].astype(float), weight=P["w"][idx].astype(float))
    seeds = common_seeds(BASE, "DY_MOMENTS", CH6)
    M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, seeds,
                                       born_tags={"mll": "mll", "y_abs": "absyz"},
                                       n_born=12, n_recoil=20,
                                       x_match=XM, x_hi=XMAP, soft_lo=SOFT)

    # fine FO histogram: reference curve AND source of the lin-map targets
    flo, fhi, fd, _ = fo_curve(BASE, "DY_MOMENTS", CH6,
                               common_seeds(BASE, "DY_MOMENTS", CH6, tag="mll_fine"),
                               "mll_fine")
    g = (fhi > flo) & np.isfinite(fd)
    flo, fhi, fd = flo[g], fhi[g], fd[g]
    inr = (flo >= A - 1e-9) & (fhi <= B + 1e-9)
    flo, fhi, fd = flo[inr], fhi[inr], fd[inr]

    # cross-check: histogram-derived BW moments vs the event-level targets
    mu_bw_hist = hist_moments(flo, fhi, fd, u_bw)
    mu_bw_ev = np.asarray(M["born"]["mll"]["values"], float)[:6]
    err_ev = np.asarray(M["born"]["mll"].get("errors", np.zeros(6)), float)[:6]
    print("BW moments, histogram-derived vs event-level (diff / FO error):")
    for k in range(6):
        d = mu_bw_hist[k] - mu_bw_ev[k]
        print(f"  n={k+1}: {mu_bw_hist[k]:+.5f} vs {mu_bw_ev[k]:+.5f}"
              f"   diff={d:+.1e}  ({abs(d)/max(err_ev[k],1e-30):.2f} sigma_FO)")

    mu_lin = hist_moments(flo, fhi, fd, u_lin)

    cfg_bw = dict(born={"mll": {"range": (A, B), "map": "bw"},
                        "y_abs": {"range": (0., 2.4), "map": "lin"}},
                  recoil={"pT_ll": {"range": (SOFT, XMAP), "map": "log", "soft_lo": SOFT,
                                    "profile": {"a": XM, "b": 2 * XM, "c": XHI}}})
    res_bw = upgrade(ev, M, cfg_bw)

    M_lin = copy.deepcopy(M)
    M_lin["born"]["mll"]["values"] = list(mu_lin)
    M_lin["born"]["mll"]["errors"] = []          # impose all six
    cfg_lin = copy.deepcopy(cfg_bw)
    cfg_lin["born"]["mll"]["map"] = "lin"
    res_lin = upgrade(ev, M_lin, cfg_lin)
    print(f"BW solve : effN {100*res_bw.effN:.1f}%  closure {res_bw.closure:.2e}")
    print(f"lin solve: effN {100*res_lin.effN:.1f}%  closure {res_lin.closure:.2e}")

    # ---- spectra on the standard 25-bin grid ------------------------------
    e = np.linspace(A, B, 26); bw = np.diff(e); ctr = 0.5 * (e[:-1] + e[1:])
    fo = rebin_density(flo, fhi, fd, e); fo = fo / (fo * bw).sum()

    def dens(w):
        h, _ = np.histogram(ev["mll"], e, weights=w)
        h = h / np.diff(e); return h / (h * np.diff(e)).sum()

    hp, hb, hl = dens(ev["weight"]), dens(res_bw.weights), dens(res_lin.weights)
    med = lambda h: 100 * np.median(np.abs(h / fo - 1))

    # uncertainties, one convention: BAND = fixed-order 7-point scale envelope
    # (shape-only, each member normalized before the envelope), BARS =
    # statistics (fixed order: seed scatter; samples: own MC error).
    fb = fo_curve_band(BASE, "DY_MOMENTS", CH6,
                       common_seeds(BASE, "DY_MOMENTS", CH6, tag="mll_fine"),
                       "mll_fine", edges=e, members=True)
    fo_lo = fo_hi = fo_st = None
    if fb is not None:
        _, _, cen, _, _, fst, mem = fb
        nrm = lambda h: h / np.nansum(h * bw)
        cen_n = nrm(cen); mem_n = np.array([nrm(m_) for m_ in mem])
        fo_lo = np.minimum(cen_n, np.nanmin(mem_n, 0)) / cen_n
        fo_hi = np.maximum(cen_n, np.nanmax(mem_n, 0)) / cen_n
        fo_st = (fst / np.nansum(cen * bw)) / cen_n

    # ratio-only: the m_ll SPECTRUM itself is a PRL figure and is not repeated
    # in the JHEP; this panel is the methods point, not the physics result
    fig, r = plt.subplots(figsize=(7.2, 4.8))
    r.stairs(hp / fo, e, color=C["prior"], ls="--", lw=1.9,
             label=rf"PS+LO prior ({med(hp):.1f}\%)")
    r.stairs(hl / fo, e, color=C["mcatnlo"], lw=2.1,
             label=rf"6 moments, linear map ({med(hl):.1f}\%)")
    r.stairs(hb / fo, e, color=C["maxent"], lw=2.8,
             label=rf"6 moments, BW map ({med(hb):.1f}\%)")
    r.axhline(1, color="k", lw=0.8)
    if fo_lo is not None:
        r.fill_between(ctr, fo_lo, fo_hi, step="mid", color=C["band"], alpha=0.55, lw=0)
        r.errorbar(stagger(e, 0, 4), np.ones_like(ctr), yerr=fo_st, fmt="none",
                   ecolor="0.35", elinewidth=1.0, capsize=1.6)
    for k_, (h_, w_, col_) in enumerate(((hp, ev["weight"], C["prior"]), (hb, res_bw.weights, C["maxent"]),
                                         (hl, res_lin.weights, C["mcatnlo"]))):
        r.errorbar(stagger(e, 1 + k_, 4), h_ / fo, yerr=mc_err(ev["mll"], w_, e) / fo, fmt="none",
                   ecolor=col_, elinewidth=1.0, capsize=1.6)
    r.set_xlabel(r"$m_{\ell\ell}$ [GeV]")
    r.set_ylabel(r"ratio to fixed order (NNLO)")
    r.set_ylim(0.90, 1.10); r.set_xlim(A, B)
    r.legend(loc="lower right", fontsize=12, labelspacing=0.28)
    out = os.path.join(HERE, "fig_map_choice.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
    print(f"medians: prior {med(hp):.1f}%  lin {med(hl):.1f}%  bw {med(hb):.1f}%")
    print("wrote", out)


if __name__ == "__main__":
    main()
