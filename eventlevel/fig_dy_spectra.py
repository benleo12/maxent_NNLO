#!/usr/bin/env python3
r"""The Drell-Yan spectra behind the PRL, one observable per figure.

A single solve produces all of them.  Constrained: the Born pair m_ll and
|y_ll|, whose moments come from the inclusive Z calculation at NNLO and are
genuinely NNLO (SNR 47-5465); plus the recoil pT_ll through the smooth profile,
which is NLO(Z+jet) accurate because pT_ll vanishes at Born level.  Predicted,
never constrained: pT_l1 and phi*_eta.

  fig_dy_mll     m_ll     constrained (NNLO)      ratio to fixed order
  fig_dy_yll     |y_ll|   constrained (NNLO)      ratio to fixed order
  fig_dy_ptl1    pT_l1    PREDICTED               ratio to PS+LO prior
  (phi* and pT_ll have their own scripts, both with ratio to ATLAS data)

The ratio denominator is the best available reference for that observable: data
where it is measured, fixed order where it is not.  pT_l1 has neither -- NNLOJET
books its moments but no histogram -- so its ratio is to the prior, which shows
the size and shape of the correction the reweighting applies; the quantitative
comparison against fixed order for pT_l1 is the pull plot in fig_nnlo_born.py.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import use_pub_style, C, LS, LW, rebin_density
use_pub_style(base=18)
from maxent_upgrade import upgrade
from nnlojet_moments import fo_moments_smooth_from_nnlojet, common_seeds, fo_curve

BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_poc"
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
XM, XHI, SOFT = 30.0, 500.0, 0.5
GENS = [("MiNNLO", "minnlo", "dy_minnlo_atlas_v2.npz"),
        ("MC@NLO", "mcatnlo", "dy_mcatnlo_atlas.npz"),
        ("POWHEG", "powheg", "dy_powheg_atlas.npz")]


def dens(x, w, e):
    h, _ = np.histogram(np.asarray(x, float), e, weights=np.asarray(w, float))
    h = h / np.asarray(w, float).sum() / np.diff(e)
    return h


def panel(name, x_key, edges, xlabel, title, fo_tag, ev, res, logx=False,
          logy=True, ratio_to="fo", ylim=(0.85, 1.15), gens=True):
    e = np.asarray(edges, float); bw = np.diff(e); ctr = 0.5 * (e[:-1] + e[1:])
    hp = dens(ev[x_key], ev["weight"], e)
    hq = dens(ev[x_key], res.weights, e)
    hp, hq = hp / (hp * bw).sum(), hq / (hq * bw).sum()

    fo = None
    if fo_tag is not None:
        fc = fo_curve(BASE, "DY_MOMENTS", CH6,
                      common_seeds(BASE, "DY_MOMENTS", CH6, tag=fo_tag), fo_tag)
        if fc is not None:
            flo, fhi, fd, _ = fc
            g = (fd > 0) & (fhi > flo)
            fo = rebin_density(flo[g], fhi[g], fd[g], e)
            ok = np.isfinite(fo) & (fo > 0)
            fo = np.where(ok, fo, np.nan); fo = fo / np.nansum(fo * bw)

    ref = fo if (ratio_to == "fo" and fo is not None) else hp
    fig, ax = plt.subplots(2, 1, figsize=(6.9, 7.4),
                           gridspec_kw={"height_ratios": [2.15, 1.1], "hspace": 0.06})
    a, r = ax
    if fo is not None:
        a.stairs(fo, e, color=C["fo"], ls=LS["fo"], lw=LW["fo"],
                 label=r"fixed order (NNLO)")
    a.stairs(hp, e, color=C["prior"], ls=LS["prior"], lw=LW["prior"], label=r"PS+LO prior")
    a.stairs(hq, e, color=C["maxent"], lw=LW["maxent"], label=r"MaxEnt")
    for lbl, key, fn in (GENS if gens else []):
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        G = dict(np.load(p))
        gk = {"pT_lead": "pT_lead", "mll": "mll", "y_ll": "y_ll", "pT_ll": "pT_ll"}.get(x_key)
        if gk is None or gk not in G:
            continue
        v = np.abs(np.asarray(G[gk], float)) if x_key == "y_ll" else np.asarray(G[gk], float)
        hg = dens(v, np.asarray(G["w"], float), e); hg = hg / (hg * bw).sum()
        a.stairs(hg, e, color=C[key], lw=LW[key], label=lbl)
        r.stairs(hg / ref, e, color=C[key], lw=1.7)
    if fo is not None and ratio_to != "fo":
        r.stairs(fo / ref, e, color=C["fo"], ls=LS["fo"], lw=2.0)
    r.stairs(hp / ref, e, color=C["prior"], ls=LS["prior"], lw=1.8)
    r.stairs(hq / ref, e, color=C["maxent"], lw=2.6)
    r.axhline(1, color="k", lw=0.8)

    if logx:
        a.set_xscale("log"); r.set_xscale("log")
    if logy:
        a.set_yscale("log")
    a.tick_params(labelbottom=False, which="both")   # minor labels too, on log axes
    a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}X$")
    a.set_title(title)
    a.legend(loc="best", fontsize=13, labelspacing=0.28)
    r.set_xlabel(xlabel); r.set_ylim(*ylim)
    r.set_ylabel(r"ratio to fixed order" if ratio_to == "fo" else r"ratio to PS+LO prior")
    a.set_xlim(e[0], e[-1]); r.set_xlim(e[0], e[-1])
    out = os.path.join(HERE, f"{name}.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
    m = np.isfinite(ref) & (ref > 0)
    print(f"  {name}: prior {100*np.median(np.abs(hp[m]/ref[m]-1)):5.1f}%   "
          f"MaxEnt {100*np.median(np.abs(hq[m]/ref[m]-1)):5.1f}%   -> {out}")


def main():
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v2.npz")))
    n = len(P["w"]); idx = np.random.default_rng(0).choice(n, min(1_000_000, n), replace=False)
    ev = dict(mll=P["mll"][idx].astype(float), y_abs=np.abs(P["y_ll"][idx]).astype(float),
              y_ll=np.abs(P["y_ll"][idx]).astype(float),
              pT_ll=P["pT_ll"][idx].astype(float),
              pT_lead=P["pT_lead"][idx].astype(float),
              phistar=P["phistar"][idx].astype(float), weight=P["w"][idx].astype(float))
    M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6,
                                       common_seeds(BASE, "DY_MOMENTS", CH6),
                                       born_tags={"mll": "mll", "y_abs": "absyz"},
                                       n_born=6, n_recoil=12,
                                       x_match=XM, x_hi=XHI, soft_lo=SOFT)
    res = upgrade(ev, M, dict(
        born={"mll": {"range": (66., 116.), "map": "lin"},
              "y_abs": {"range": (0., 2.4), "map": "lin"}},
        recoil={"pT_ll": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                          "profile": {"a": XM, "b": 2 * XM, "c": XHI}}},
        followers=["phistar", "pT_lead"]))
    print(f"solve: effN {100*res.effN:.1f}%  closure {res.closure:.2e}  "
          f"neg-wt {100*np.mean(res.weights<=0):.1f}%\n")

    # No generators here.  The showered samples carry QED final-state radiation
    # and the fixed-order calculation does not, so their low-mass tails differ
    # by a physics effect that has nothing to do with the reweighting; putting
    # them on the same axes would read as disagreement.  Twenty-five bins, not
    # fifty: six Chebyshev moments cannot resolve a Breit-Wigner more finely
    # than that, and over-resolving it only shows the fit oscillating.
    panel("fig_dy_mll", "mll", np.linspace(66, 116, 26), r"$m_{\ell\ell}$ [GeV]",
          r"$m_{\ell\ell}$, constrained at NNLO", "mll_fine", ev, res,
          logy=True, ylim=(0.80, 1.20), gens=False)
    panel("fig_dy_yll", "y_ll", np.linspace(0.0, 2.4, 25), r"$|y_{\ell\ell}|$",
          r"$|y_{\ell\ell}|$, constrained at NNLO", "absyz_fine", ev, res,
          logy=False, ylim=(0.85, 1.15))
    panel("fig_dy_ptl1", "pT_lead", np.geomspace(27, 200, 26), r"$p_T^{\ell_1}$ [GeV]",
          r"$p_T^{\ell_1}$, never constrained", None, ev, res,
          logx=True, logy=True, ratio_to="prior", ylim=(0.7, 2.0))


if __name__ == "__main__":
    main()
