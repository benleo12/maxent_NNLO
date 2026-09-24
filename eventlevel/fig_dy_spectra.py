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
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import use_pub_style, C, LS, LW, rebin_density, gen_scale_weights
use_pub_style(base=18)
from maxent_upgrade import upgrade
from nnlojet_moments import (fo_moments_smooth_from_nnlojet, common_seeds,
                             fo_curve, fo_curve_band)

BASE = os.path.join(os.environ.get("NNLOJET_ROOT",
        os.path.expanduser("~/nnlojet-v1.0.2")), "dy_profile_log30_hi")
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
XM, XHI, SOFT, XMAP = 30.0, 500.0, 30.0, 2500.0
GENS = [("MiNNLO$_{\\mathrm{PS}}$", "minnlo", "dy_minnlo_atlas_v4.npz"),
        ("MC@NLO", "mcatnlo", "dy_mcatnlo_atlas_v4.npz"),
        ("POWHEG", "powheg", "dy_powheg_atlas_v6.npz")]


def dens(x, w, e):
    h, _ = np.histogram(np.asarray(x, float), e, weights=np.asarray(w, float))
    h = h / np.asarray(w, float).sum() / np.diff(e)
    return h


def panel(name, x_key, edges, xlabel, title, fo_tag, ev, res, logx=False,
          logy=True, ratio_to="fo", ylim=(0.85, 1.15), gens=True, unc=None,
          res2=None, unc2=None):
    e = np.asarray(edges, float); bw = np.diff(e); ctr = 0.5 * (e[:-1] + e[1:])
    hp = dens(ev[x_key], ev["weight"], e)
    hq = dens(ev[x_key], res.weights, e)
    hp, hq = hp / (hp * bw).sum(), hq / (hq * bw).sum()

    def nd(w):
        h = dens(ev[x_key], w, e); return h / (h * bw).sum()

    # ---- MaxEnt uncertainty components from the variant weight vectors ----
    # scale: per-bin envelope over the 6 warm-started scale re-solves
    # stat : bootstrap-over-seeds spread (+) the sample's own MC error
    q_lo = q_hi = q_err = q_scale_half = None
    if unc is not None:
        hs = np.array([nd(w) for w in unc["scale_w"]])
        boot = np.array([nd(w) for w in unc["boot_w"]]).std(0, ddof=1)
        qq = res.weights / res.weights.sum()
        s2, _ = np.histogram(ev[x_key], e, weights=qq * qq)
        mc = np.sqrt(s2) / bw / (dens(ev[x_key], res.weights, e) * bw).sum()
        q_err = np.hypot(boot, mc)
        # LIKE FOR LIKE with the fixed-order band.  The grey band is the
        # reference's scale envelope combined with its seed scatter, i.e. its
        # total uncertainty; the upgraded sample's band must therefore also be
        # its total -- scale envelope of the target moments, combined with the
        # moments' Monte Carlo error through the bootstrap re-solves and with
        # the sample's own finite statistics.  Drawing scale alone here (as an
        # earlier version did) compares a scale-only band against a
        # scale-plus-statistics one and makes the upgrade look six times more
        # certain than the input it is built from.  Scale against scale the two
        # agree to a factor two, as they must: the sample inherits its scale
        # dependence entirely from the moments, and reproduces the part of the
        # reference's scale variation that the imposed tower can represent.
        q_lo = np.minimum(hq, hs.min(0))
        q_hi = np.maximum(hq, hs.max(0))
        q_scale_half = 0.5 * (q_hi - q_lo)
    # the same upgrade with the NNLO Z+jet recoil (Stripper), same treatment
    hq2 = nd(res2.weights) if res2 is not None else None
    q2_lo = q2_hi = q2_err = None
    if hq2 is not None and unc2 is not None:
        hs2 = np.array([nd(w) for w in unc2["scale_w"]])
        boot2 = np.array([nd(w) for w in unc2["boot_w"]]).std(0, ddof=1)
        qq2 = res2.weights / res2.weights.sum()
        s2b, _ = np.histogram(ev[x_key], e, weights=qq2 * qq2)
        mc2 = np.sqrt(s2b) / bw / (dens(ev[x_key], res2.weights, e) * bw).sum()
        q2_err = np.hypot(boot2, mc2)
        q2_lo = np.minimum(hq2, hs2.min(0)); q2_hi = np.maximum(hq2, hs2.max(0))

    fo = fo_lo = fo_hi = fo_st = None
    if fo_tag is not None:
        fb = fo_curve_band(BASE, "DY_MOMENTS", CH6,
                           common_seeds(BASE, "DY_MOMENTS", CH6, tag=fo_tag),
                           fo_tag, edges=e, members=True)
        if fb is not None:
            _, _, fd, _, _, fst_, allc = fb
            ok = np.isfinite(fd) & (fd > 0)
            nrm = np.nansum(np.where(ok, fd, np.nan) * bw)
            fo = np.where(ok, fd, np.nan) / nrm
            # SHAPE-ONLY scale band: this panel is a normalized density, on
            # which the MaxEnt sample has no rate freedom (sum q = 1), so the
            # like-for-like FO band normalizes each scale member to unit
            # integral BEFORE the envelope.  Normalizing the envelope curves
            # instead inflates the band by the window-rate variation.
            mem = np.array([np.where(ok, c, np.nan) for c in allc])
            mem = mem / np.nansum(mem * bw, axis=1, keepdims=True)
            fo_lo = np.minimum(fo, np.nanmin(mem, 0))
            fo_hi = np.maximum(fo, np.nanmax(mem, 0))
            fo_st = fst_ / nrm

    ref = fo if (ratio_to == "fo" and fo is not None) else hp
    # resolve the generator list FIRST so the stat-bar staggering knows how
    # many series share the ratio panel
    gen_list = []
    for lbl, key, fn in (GENS if gens else []):
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        G = dict(np.load(p))
        gk = {"pT_lead": "pT_lead", "mll": "mll", "y_ll": "y_ll", "pT_ll": "pT_ll"}.get(x_key)
        if gk is None or gk not in G:
            continue
        gen_list.append((lbl, key, G, gk))
    NSER = 4 + len(gen_list)
    from bandviz import stagger

    def stat_bars(x_, w_, h_, col_, k_, extra=None, axr=None):
        # per-bin MC error of the normalized density, staggered per series
        ww = np.asarray(w_, float)
        s2, _ = np.histogram(x_, e, weights=(ww / ww.sum()) ** 2)
        err = np.sqrt(s2) / bw
        if extra is not None:
            err = np.hypot(err, extra)
        (axr if axr is not None else r).errorbar(stagger(e, k_, NSER), h_ / ref, yerr=err / ref, fmt="none",
                   ecolor=col_, elinewidth=1.1, capsize=1.8)

    # Two ratio panels to the same reference: (r) the upgrade against its
    # prior and the fixed order it was built from; (r2) the upgrade against
    # the matched generators.  Same reference, same bands, same candles.
    fig, ax = plt.subplots(3, 1, figsize=(6.9, 9.6),
                           gridspec_kw={"height_ratios": [2.15, 1.0, 1.0],
                                        "hspace": 0.06})
    a, r, r2 = ax
    if fo is not None:
        a.stairs(fo, e, color=C["fo"], ls=LS["fo"], lw=LW["fo"],
                 label=r"fixed order (NNLO)")
    a.stairs(hp, e, color=C["prior"], ls=LS["prior"], lw=LW["prior"], label=r"PS+LO prior")
    a.stairs(hq, e, color=C["maxent"], lw=LW["maxent"],
             label=(r"MaxEnt, NLO$(Zj)$ recoil" if hq2 is not None else r"MaxEnt"))
    if hq2 is not None:
        a.stairs(hq2, e, color=C["maxent_nnlo"], lw=LW["maxent_nnlo"], label=r"MaxEnt, NNLO$(Zj)$ recoil")
    gen_ratios = []
    for kg, (lbl, key, G, gk) in enumerate(gen_list):
        v = np.abs(np.asarray(G[gk], float)) if x_key == "y_ll" else np.asarray(G[gk], float)
        gw = np.asarray(G["w"], float)
        hg = dens(v, gw, e); hg = hg / (hg * bw).sum()
        a.stairs(hg, e, color=C[key], lw=LW[key], label=lbl)
        r2.stairs(hg / ref, e, color=C[key], lw=1.7)
        gen_ratios.append(hg)
        # this sample's own 7-point scale envelope + MC stat bars
        if "w_scale" in G:
            hgs = []; WS = gen_scale_weights(G)
            for kk in range(WS.shape[1]):
                h_ = dens(v, WS[:, kk], e)
                hgs.append(h_ / (h_ * bw).sum())
            hgs = np.array(hgs)
            r2.fill_between(ctr, np.minimum(hg, hgs.min(0)) / ref,
                            np.maximum(hg, hgs.max(0)) / ref,
                            step="mid", color=C[key], alpha=0.13, lw=0)
        stat_bars(v, gw, hg, C[key], 4 + kg, axr=r2)
    if fo is not None and ratio_to != "fo":
        for rr_ in (r, r2):
            rr_.stairs(fo / ref, e, color=C["fo"], ls=LS["fo"], lw=2.0)
    # ---- uncertainty visuals: ONE convention, applied to every curve ----
    # BAND  = scale uncertainty (7-point envelope)
    # CANDLE = statistical uncertainty (Monte Carlo / bootstrap)
    # for the fixed order, the upgraded sample, the prior and the generators
    # alike, so that any two curves' scale uncertainties may be compared with
    # each other and any two statistical ones likewise.  Mixing the two into a
    # single band, as an earlier version did, hides which is which and makes
    # curves with different Monte Carlo sizes look systematically different.
    for rr_ in (r, r2):
        if ratio_to == "fo" and fo_lo is not None:
            rr_.fill_between(ctr, fo_lo / ref, fo_hi / ref, color=C["band"],
                             alpha=0.55, step="mid", lw=0, label=r"FO scale")
            for s_, edge in ((-1, fo_lo), (1, fo_hi)):   # readable under overlap
                rr_.plot(ctr, edge / ref, color="0.30", lw=0.8,
                         drawstyle="steps-mid", zorder=3.5)
            rr_.errorbar(stagger(e, 0, NSER), np.ones_like(ctr),
                         yerr=fo_st / np.where(ref > 0, ref, np.inf), fmt="none",
                         ecolor="0.35", elinewidth=1.1, capsize=1.8)
        if q_lo is not None:
            rr_.fill_between(ctr, q_lo / ref, q_hi / ref, color=C["maxent"],
                             alpha=0.25, step="mid", lw=0)
        rr_.stairs(hq / ref, e, color=C["maxent"], lw=2.6)
        if q_err is not None:
            rr_.errorbar(stagger(e, 2, NSER), hq / ref,
                         yerr=q_err / np.where(ref > 0, ref, np.inf), fmt="none",
                         ecolor=C["maxent"], elinewidth=1.3, capsize=2.2)
        if hq2 is not None:
            if q2_lo is not None:
                rr_.fill_between(ctr, q2_lo / ref, q2_hi / ref, color=C["maxent_nnlo"],
                                 alpha=0.25, step="mid", lw=0)
            rr_.stairs(hq2 / ref, e, color=C["maxent_nnlo"], lw=2.6)
            if q2_err is not None:
                rr_.errorbar(stagger(e, 3, NSER), hq2 / ref,
                             yerr=q2_err / np.where(ref > 0, ref, np.inf), fmt="none",
                             ecolor=C["maxent_nnlo"], elinewidth=1.3, capsize=2.2)
        rr_.axhline(1, color="k", lw=0.8)
    r.stairs(hp / ref, e, color=C["prior"], ls=LS["prior"], lw=1.8)
    stat_bars(ev[x_key], ev["weight"], hp, C["prior"], 1)

    if logx:
        a.set_xscale("log"); r.set_xscale("log"); r2.set_xscale("log")
    if logy:
        a.set_yscale("log")
    a.tick_params(labelbottom=False, which="both")   # minor labels too, on log axes
    r.tick_params(labelbottom=False, which="both")
    a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}X$")
    a.set_title(title)
    a.legend(loc="best", fontsize=13, labelspacing=0.28)
    r2.set_xlabel(xlabel); r.set_ylim(*ylim); r2.set_ylim(*ylim)
    rl = r"ratio to NNLO" if ratio_to == "fo" else r"ratio to prior"
    r.set_ylabel(rl); r2.set_ylabel(rl)
    r.text(0.02, 0.9, "upgrade vs prior", transform=r.transAxes, fontsize=11, va="top")
    r2.text(0.02, 0.9, "upgrade vs matched generators", transform=r2.transAxes, fontsize=11, va="top")
    a.set_xlim(e[0], e[-1]); r.set_xlim(e[0], e[-1]); r2.set_xlim(e[0], e[-1])

    out = os.path.join(HERE, f"{name}.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
    m = np.isfinite(ref) & (ref > 0)
    print(f"  {name}: prior {100*np.median(np.abs(hp[m]/ref[m]-1)):5.1f}%   "
          f"MaxEnt {100*np.median(np.abs(hq[m]/ref[m]-1)):5.1f}%   "
          + (f"MaxEnt NNLO(Zj) recoil {100*np.median(np.abs(hq2[m]/ref[m]-1)):5.1f}%   " if hq2 is not None else "")
          + f"-> {out}")


def main():
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v3.npz")))
    n = len(P["w"]); idx = np.random.default_rng(0).choice(n, min(1_000_000, n), replace=False)
    ev = dict(mll=P["mll"][idx].astype(float), y_abs=np.abs(P["y_ll"][idx]).astype(float),
              y_ll=np.abs(P["y_ll"][idx]).astype(float),
              pT_ll=P["pT_ll"][idx].astype(float),
              pT_lead=P["pT_lead"][idx].astype(float),
              phistar=P["phistar"][idx].astype(float), weight=P["w"][idx].astype(float))
    M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6,
                                       common_seeds(BASE, "DY_MOMENTS", CH6),
                                       born_tags={"mll": "mll", "y_abs": "absyz"},
                                       n_born=12, n_recoil=20,
                                       x_match=XM, x_hi=XMAP, soft_lo=SOFT)
    res = upgrade(ev, M, dict(
        born={"mll": {"range": (66., 116.), "map": "bw"},
              "y_abs": {"range": (0., 2.4), "map": "lin"}},
        recoil={"pT_ll": {"range": (SOFT, XMAP), "map": "log", "soft_lo": SOFT,
                          "profile": {"a": XM, "b": 2 * XM, "c": XHI}}},
        followers=["phistar", "pT_lead"]))
    print(f"solve: effN {100*res.effN:.1f}%  closure {res.closure:.2e}  "
          f"neg-wt {100*np.mean(res.weights<=0):.1f}%\n")
    res2 = None
    xml = os.path.join(HERE, "ppzj-moments_NNLO.xml")
    if os.path.exists(xml):
        os.environ["DY_RECOIL_XML"] = xml
        try:
            M2 = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6,
                                                common_seeds(BASE, "DY_MOMENTS", CH6),
                                                born_tags={"mll": "mll", "y_abs": "absyz"},
                                                n_born=12, n_recoil=20,
                                                x_match=XM, x_hi=XMAP, soft_lo=SOFT)
        finally:
            os.environ.pop("DY_RECOIL_XML", None)
        res2 = upgrade(ev, M2, dict(
            born={"mll": {"range": (66., 116.), "map": "bw"},
                  "y_abs": {"range": (0., 2.4), "map": "lin"}},
            recoil={"pT_ll": {"range": (SOFT, XMAP), "map": "log", "soft_lo": SOFT,
                              "profile": {"a": XM, "b": 2 * XM, "c": XHI}}},
            followers=["phistar", "pT_lead"]))
        print(f"solve with the NNLO(Zj) recoil: effN {100*res2.effN:.1f}%  closure {res2.closure:.2e}\n")

    # variant weight vectors from moment_bands (6 scale re-solves + bootstrap);
    # produced on the SAME rng(0) event subsample, verified via the stored idx
    unc = None
    bwf = os.path.join(HERE, "dy_band_weights.npz")
    if os.path.exists(bwf):
        BW = dict(np.load(bwf))
        if np.array_equal(BW["idx"], idx) and np.allclose(BW["central"], res.weights,
                                                          rtol=1e-6, atol=0):
            unc = {"scale_w": BW["scale_w"], "boot_w": BW["boot_w"]}
            print("  uncertainty variants loaded (6 scale + "
                  f"{len(BW['boot_w'])} bootstrap)")
        else:
            print("  WARNING: dy_band_weights.npz does not match this solve -- "
                  "re-run moment_bands; drawing without bands")
    unc2 = None
    bwf2 = os.path.join(HERE, "dy_band_weights_nnloZj.npz")
    if res2 is not None and os.path.exists(bwf2):
        BW2 = dict(np.load(bwf2))
        if np.array_equal(BW2["idx"], idx) and np.allclose(BW2["central"], res2.weights, rtol=1e-6, atol=0):
            unc2 = {"scale_w": BW2["scale_w"], "boot_w": BW2["boot_w"]}
            print(f"  NNLO(Zj) variants loaded (6 scale + {len(BW2['boot_w'])} bootstrap)")
        else:
            print("  WARNING: dy_band_weights_nnloZj.npz does not match this solve; NNLO(Zj) curve without bands")

    # No generators here.  The showered samples carry QED final-state radiation
    # and the fixed-order calculation does not, so their low-mass tails differ
    # by a physics effect that has nothing to do with the reweighting; putting
    # them on the same axes would read as disagreement.  Twenty-five bins, not
    # fifty: six Chebyshev moments cannot resolve a Breit-Wigner more finely
    # than that, and over-resolving it only shows the fit oscillating.
    panel("fig_dy_mll", "mll", np.linspace(66, 116, 26), r"$m_{\ell\ell}$ [GeV]",
          r"$m_{\ell\ell}$, constrained at NNLO", "mll_fine", ev, res, unc=unc,
          logy=True, ylim=(0.80, 1.20), gens=True, res2=res2, unc2=unc2)
    panel("fig_dy_yll", "y_ll", np.linspace(0.0, 2.4, 25), r"$|y_{\ell\ell}|$",
          r"$|y_{\ell\ell}|$, constrained at NNLO", "absyz_fine", ev, res, unc=unc,
          logy=False, ylim=(0.85, 1.15), res2=res2, unc2=unc2)
    panel("fig_dy_ptl1", "pT_lead", np.geomspace(27, 200, 26), r"$p_T^{\ell_1}$ [GeV]",
          r"$p_T^{\ell_1}$, never constrained", "ptl1_a", ev, res, unc=unc,
          logx=True, logy=True, ratio_to="fo", ylim=(0.6, 1.6), res2=res2, unc2=unc2)


if __name__ == "__main__":
    main()
