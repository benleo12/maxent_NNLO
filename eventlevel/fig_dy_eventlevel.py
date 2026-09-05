#!/usr/bin/env python3
r"""The main Drell-Yan figure, rebuilt from EVENT-LEVEL moments in the common style.

Replaces the stale FIG_atlas_nnlo / fig_dy_nnlo / fig_compare5: ATLAS 1912.02844
data, the LO+PS prior, the MaxEnt upgrade (event-level NNLO moments, 0% negative
weights) and every matched generator we have (MiNNLO$_{\\mathrm{PS}}$, MC@NLO, POWHEG), for the
constrained recoil pT_ll and the unconstrained follower phi*.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import use_pub_style, C, LW, rebin_density
use_pub_style(base=17)
from maxent_upgrade import upgrade, check_seam
from nnlojet_moments import (fo_moments_smooth_from_nnlojet, common_seeds,
                             _load, oriented_fo_curve, fo_curve_band)

BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_poc"
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
# mirrors eval_w_ptz (pa=30, pb=60, pc=pd=500)
XM, XHI, SOFT = 30.0, 500.0, 0.5
# Upper edge of the pT_ll panel.  Every displayed bin then lies fully
# inside the constraint window [XM, XHI], carries >=31 prior events
# (<18% Monte Carlo error) and <4% data uncertainty.  Above it every
# sample on the figure -- ours and the matched generators alike -- is
# statistics-limited, so the comparison stops being about physics.
# The medians are insensitive to the choice: cutting at 300/400/470/
# 2500 GeV gives MaxEnt 4.20/4.48/4.71/5.72% against prior
# 7.68/10.07/10.10/11.04%.
PLOT_HI = 400.0
Q_HARD = 91.1876

GENS = [("MiNNLO$_{\\mathrm{PS}}$", "dy_minnlo_atlas_v3.npz", C["minnlo"], 23),
        ("MC@NLO", "dy_mcatnlo_atlas_v4.npz", C["mcatnlo"], 5),
        ("POWHEG", "dy_powheg_atlas_v4.npz", C["powheg"], 1)]
# ONE observable per figure.  phi* has its own figure (fig_phistar_prediction);
# showing it here as well duplicated the same curves in two places.
PANELS = [("pT_ll", "atlas_pTll_born.npz", r"$p_T^{\ell\ell}$ [GeV]", "constrained")]
# pT_ll > 0 REQUIRES a real emission, so the inclusive Z calculation at NNLO
# gives this observable at NLO(Z+jet) accuracy.  m_ll and |y_ll|, the Born
# observables that are also constrained, are the NNLO ones.
FO_LABEL = r"fixed order (NLO $Z$+jet)"


def dens(x, w, e):
    h, _ = np.histogram(x, e, weights=w / w.sum()); return h / np.diff(e)


def main():
    check_seam(XM, Q_HARD, label="Drell-Yan")
    seeds = common_seeds(BASE, "DY_MOMENTS", CH6)
    print(f"DY FO seeds usable: {seeds}")
    M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, seeds,
                                       born_tags={"mll": "mll", "y_abs": "absyz"},
                                       n_born=6, n_recoil=12,
                                       x_match=XM, x_hi=XHI, soft_lo=SOFT)
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v3.npz")))
    # SAME 1M rng(0) subsample as every other DY figure and as the
    # dy_band_weights.npz variants -- a different subsample here would make
    # the shared uncertainty vectors unusable and the medians subtly figure-
    # dependent.
    n = len(P["w"]); idx = np.random.default_rng(0).choice(n, min(1_000_000, n), replace=False)
    ev = dict(mll=P["mll"][idx].astype(float), y_abs=np.abs(P["y_ll"][idx]).astype(float),
              pT_ll=P["pT_ll"][idx].astype(float), phistar=P["phistar"][idx].astype(float),
              weight=P["w"][idx].astype(float))
    cfg = dict(born={"mll": {"range": (66., 116.), "map": "bw"},
                     "y_abs": {"range": (0., 2.4), "map": "lin"}},
               recoil={"pT_ll": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                                 "profile": {"a": XM, "b": 2 * XM, "c": XHI}}},
               followers=["phistar"])
    print("solving ...", flush=True)
    res = upgrade(ev, M, cfg)
    print(f"  effN {100*res.effN:.1f}%  closure {res.closure:.2e}  "
          f"neg-wt {100*np.mean(res.weights<=0):.1f}%")

    # uncertainty variants: MaxEnt scale re-solves + bootstrap (moment_bands)
    from bandviz import stagger
    mx_scale_w = mx_boot_w = None
    bwf = os.path.join(HERE, "dy_band_weights.npz")
    if os.path.exists(bwf):
        BW = dict(np.load(bwf))
        if np.allclose(BW["central"], res.weights, rtol=1e-6, atol=0):
            mx_scale_w, mx_boot_w = BW["scale_w"], BW["boot_w"]
        else:
            print("  WARNING: dy_band_weights.npz stale; no MaxEnt bands")

    # Two ratio panels to the data: (r) the upgrade against its prior and the
    # fixed-order input; (r2) the upgrade against the matched generators.
    fig, ax = plt.subplots(3, len(PANELS), figsize=(6.9 * len(PANELS), 10.6), squeeze=False,
                           gridspec_kw={"height_ratios": [2.2, 1.0, 1.0], "hspace": 0.07,
                                        "wspace": 0.24})
    summary = {}
    for j, (key, dfile, lab, role) in enumerate(PANELS):
        a, r, r2 = ax[0, j], ax[1, j], ax[2, j]
        D = dict(np.load(os.path.join(HERE, dfile)))
        lo, hi = np.asarray(D["lo"], float), np.asarray(D["hi"], float)
        val, err = np.asarray(D["val"], float), np.asarray(D["err"], float)
        e = np.concatenate([lo[:1], hi]); ctr = np.asarray(D["center"], float)
        # Show only where the method makes a claim.  The recoil window closes
        # at XHI, so above it no fixed-order information is imposed at all and
        # the sample is the untouched prior; the prior also runs out of events
        # there (6 above 470 GeV, none above 650 in 1M).  Plotting further was
        # showing pure Monte Carlo noise inside a region we never constrain.
        m = val > 0
        if key == "pT_ll":
            m &= hi <= PLOT_HI
        med = lambda h: 100 * np.median(np.abs(h[m] / val[m] - 1))

        a.errorbar(ctr[m], val[m], yerr=err[m], fmt="o", color="k", ms=4, lw=1.1,
                   label=r"ATLAS 1912.02844", zorder=10)
        hp = dens(ev[key], ev["weight"], e)
        hq = dens(ev[key], res.weights, e)
        a.stairs(np.where(m, hp, np.nan), e, color=C["prior"], ls="--", lw=2.0,
                 label=rf"PS+LO prior ({med(hp):.1f}\%)")
        a.stairs(np.where(m, hq, np.nan), e, color=C["maxent"], lw=3.2,
                 label=rf"MaxEnt ({med(hq):.1f}\%, $0\%\,w<0$)")
        r.stairs(np.where(m, hp / np.maximum(val, 1e-30), np.nan), e, color=C["prior"], ls="--", lw=1.8)
        for rr_ in (r, r2):
            rr_.stairs(np.where(m, hq / np.maximum(val, 1e-30), np.nan), e, color=C["maxent"], lw=2.6)
        summary[key] = {"prior": med(hp), "MaxEnt": med(hq)}
        vref = np.maximum(val, 1e-30)
        NSER = 5   # prior, MaxEnt, 3 generators -- for bar staggering

        def stat_bars(x_, w_, h_, col_, k_, extra=None, axes=None):
            ww = np.asarray(w_, float)
            s2, _ = np.histogram(x_, e, weights=(ww / ww.sum()) ** 2)
            err = np.sqrt(s2) / np.diff(e)
            if extra is not None:
                err = np.hypot(err, extra)
            for ax_ in (axes if axes is not None else (r,)):
                ax_.errorbar(stagger(e, k_, NSER), np.where(m, h_ / vref, np.nan),
                             yerr=np.where(m, err / vref, np.nan), fmt="none",
                             ecolor=col_, elinewidth=1.1, capsize=1.8)

        stat_bars(ev[key], ev["weight"], hp, C["prior"], 0)
        if mx_scale_w is not None:
            # BAND = scale envelope, CANDLE = statistics (bootstrap (+) own
            # MC): the one convention used by every curve in these figures.
            hs = np.array([dens(ev[key], w_, e) for w_ in mx_scale_w])
            lo_, hi_ = np.minimum(hq, hs.min(0)), np.maximum(hq, hs.max(0))
            for rr_ in (r, r2):
                rr_.fill_between(ctr, np.where(m, lo_ / vref, np.nan),
                                 np.where(m, hi_ / vref, np.nan),
                                 step="mid", color=C["maxent"], alpha=0.15, lw=0)
            boot = np.array([dens(ev[key], w_, e)
                             for w_ in mx_boot_w]).std(0, ddof=1)
            stat_bars(ev[key], res.weights, hq, C["maxent"], 1, extra=boot, axes=(r, r2))
        for kg, (lbl, fn, col, neg) in enumerate(GENS):
            p = os.path.join(HERE, fn)
            if not os.path.exists(p):
                continue
            G = dict(np.load(p))
            if key not in G:
                continue
            gx, gw = np.asarray(G[key], float), np.asarray(G["w"], float)
            hg = dens(gx, gw, e)
            a.stairs(np.where(m, hg, np.nan), e, color=col, lw=1.9,
                     label=rf"{lbl} ({med(hg):.1f}\%, ${neg}\%\,w<0$)")
            r2.stairs(np.where(m, hg / vref, np.nan), e, color=col, lw=1.7)
            if "w_scale" in G:
                hgs = np.array([dens(gx, G["w_scale"][:, kk].astype(float), e)
                                for kk in range(G["w_scale"].shape[1])])
                r2.fill_between(ctr, np.where(m, np.minimum(hg, hgs.min(0)) / vref, np.nan),
                                np.where(m, np.maximum(hg, hgs.max(0)) / vref, np.nan),
                                step="mid", color=col, alpha=0.12, lw=0)
            stat_bars(gx, gw, hg, col, 2 + kg, axes=(r2,))
            summary[key][lbl] = med(hg)
        # FIXED ORDER: phi* is booked too (phistar_a).  It is a prediction only
        # ABOVE the seam -- below it the Sudakov logarithms are unresummed --
        # so draw it faded there rather than let its spikes dominate the panel.
        if key == "phistar":
            fc = oriented_fo_curve(BASE, "DY_MOMENTS", CH6,
                          common_seeds(BASE, "DY_MOMENTS", CH6, tag="phistar_a"),
                          "phistar_a")
            if fc is not None:
                flo, fhi, fd, _ = fc
                g = (fd > 0) & (fhi > flo)
                # SAME EDGES as data / prior / MaxEnt / generators on this panel
                fo = rebin_density(flo[g], fhi[g], fd[g], e)
                gd = np.isfinite(fo) & (fo > 0)
                fn = np.where(gd, fo, np.nan)
                fn = fn / np.nansum(fn * np.diff(e))
                xs_ = XM / 91.1876
                ab = ctr >= xs_
                a.stairs(np.where(ab, fn, np.nan), e, color=C["fo"], ls=":", lw=LW["fo"],
                         label=FO_LABEL)
                a.stairs(np.where(~ab, fn, np.nan), e, color=C["fo"], ls=":", lw=1.4, alpha=0.22)
                rr = fn / np.maximum(val, 1e-30)
                for rr_ in (r, r2):
                    rr_.stairs(np.where(m & ab, rr, np.nan), e, color=C["fo"], ls=":", lw=2.2)
                    rr_.stairs(np.where(m & ~ab, rr, np.nan), e, color=C["fo"], ls=":",
                               lw=1.3, alpha=0.22)
        if key == "pT_ll":
            # Full uncertainty treatment, same as every other curve on the
            # panel: scale band = 7-point envelope, (+) seed scatter.  The
            # anchor (MaxEnt window rate) is a common factor on all variants,
            # so the band shows the FO window spectrum's own scale variation.
            fb = fo_curve_band(BASE, "DY_MOMENTS", ("R", "RR", "RV"),
                               common_seeds(BASE, "DY_MOMENTS", ("R", "RR", "RV"),
                                            tag="ptz_winfine"),
                               "ptz_winfine", edges=e)
            if fb is not None:
                _, _, fd, fsl, fsh, fst = fb
                inw = (ev[key] >= XM) & (ev[key] < XHI)
                tgt = float(res.weights[inw].sum() / res.weights.sum())
                gd = np.isfinite(fd) & (fd > 0)
                fn = np.where(gd, fd, np.nan)
                k_ = tgt / np.nansum(fn * np.diff(e))
                fn, fsl, fsh, fst = fn * k_, fsl * k_, fsh * k_, fst * k_
                a.stairs(fn, e, color=C["fo"], ls=":", lw=LW["fo"],
                         label=FO_LABEL)
                rr_ = fn / vref
                rb_ = (0.5 * (fsh - fsl)) / vref          # BAND = scale only
                rs_ = fst / vref                          # CANDLE = seed scatter
                for ax_ in (r, r2):
                    ax_.errorbar(0.5 * (e[:-1] + e[1:]), np.where(m, rr_, np.nan),
                                 yerr=np.where(m, rs_, np.nan), fmt="none",
                                 ecolor=C["fo"], elinewidth=1.0, capsize=1.6, alpha=0.8)
                    ax_.stairs(np.where(m, rr_, np.nan), e,
                               color=C["fo"], ls=":", lw=2.0)
                    ax_.fill_between(ctr, np.where(m, rr_ - rb_, np.nan),
                                     np.where(m, rr_ + rb_, np.nan),
                                     step="mid", color=C["fo"], alpha=0.15, lw=0)
        rel = err / np.maximum(val, 1e-30)
        for rr_ in (r, r2):
            rr_.fill_between(ctr[m], (1 - rel)[m], (1 + rel)[m], color=C["band"], alpha=0.55, step="mid")
            rr_.axhline(1, color="k", lw=0.8)
        if key == "pT_ll":
            for p_ in (a, r, r2):
                p_.axvspan(XM, XHI, color="#ffd24d", alpha=0.13)
                p_.axvline(XM, color=C["seam"], lw=2.0, ls="--")
        else:      # phi* : the seam maps over as phi* ~ pT/m
            for p_ in (a, r, r2):
                p_.axvline(XM / 91.1876, color=C["seam"], lw=2.0, ls="--")
        a.set_xscale("log"); a.set_yscale("log"); r.set_xscale("log"); r2.set_xscale("log")
        x0 = e[e > 0].min()
        x1 = PLOT_HI if key == "pT_ll" else e[-1]
        a.set_xlim(x0, x1); r.set_xlim(x0, x1); r2.set_xlim(x0, x1)
        a.tick_params(labelbottom=False); r.tick_params(labelbottom=False)
        r.set_ylim(0.72, 1.28); r2.set_ylim(0.72, 1.28)
        # one clear title, no subtitles -- same convention as every other figure
        a.set_title(rf"$p_T^{{\ell\ell}}$, {role} above the seam")
        r2.set_xlabel(lab)
        r.text(0.02, 0.92, "upgrade vs prior and fixed order", transform=r.transAxes, fontsize=11, va="top")
        r2.text(0.02, 0.92, "upgrade vs matched generators", transform=r2.transAxes, fontsize=11, va="top")
        if j == 0:
            a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}X$")
            r.set_ylabel(r"ratio to data"); r2.set_ylabel(r"ratio to data")
        a.legend(loc="lower left", fontsize=11.5, labelspacing=0.28)
    out = os.path.join(HERE, "fig_dy_eventlevel.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    print("wrote", out)
    for k, v in summary.items():
        print(f"  {k}: " + "  ".join(f"{a_}={b_:.2f}%" for a_, b_ in v.items()))


if __name__ == "__main__":
    main()
