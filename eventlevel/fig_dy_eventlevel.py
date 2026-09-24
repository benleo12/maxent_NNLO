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
from pubstyle import use_pub_style, C, LW, rebin_density, gen_scale_weights
use_pub_style(base=17)
from maxent_upgrade import upgrade, check_seam
from nnlojet_moments import (fo_moments_smooth_from_nnlojet, common_seeds,
                             _load, oriented_fo_curve, fo_curve_band)

BASE = os.path.join(os.environ.get("NNLOJET_ROOT",
        os.path.expanduser("~/nnlojet-v1.0.2")), "dy_profile_log30_hi")
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
# mirrors eval_w_ptz (pa=30, pb=60, pc=pd=500)
XM, XHI, SOFT, XMAP = 30.0, 500.0, 30.0, 2500.0
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

GENS = [("MiNNLO$_{\\mathrm{PS}}$", "dy_minnlo_atlas_v4.npz", C["minnlo"], 23),
        ("MC@NLO", "dy_mcatnlo_atlas_v4.npz", C["mcatnlo"], 5),
        ("POWHEG", "dy_powheg_atlas_v6.npz", C["powheg"], 1)]
# ONE observable per figure.  phi* has its own figure (fig_phistar_prediction);
# showing it here as well duplicated the same curves in two places.
PANELS = [("pT_ll", "atlas_pTll_born.npz", r"$p_T^{\ell\ell}$ [GeV]", "constrained")]
# pT_ll > 0 REQUIRES a real emission, so the inclusive Z calculation at NNLO
# gives this observable at NLO(Z+jet) accuracy.  m_ll and |y_ll|, the Born
# observables that are also constrained, are the NNLO ones.
FO_LABEL = r"fixed order (NLO $Z$+jet)"
# The NNLO Z+jet recoil (Stripper, R. Poncelet): the same Born towers, the pT_ll
# tower replaced through the DY_RECOIL_XML hook of nnlojet_moments.  Its nine
# moments (1-7% each) build the second MaxEnt curve; its direct pT_Z histogram
# (ATLAS edges, 10-20% per bin below 85 GeV, 2-7% above) is drawn as the second
# fixed-order curve from the seam up, his generation cut being pT_Z > 30 GeV.
NNLO_XML = os.path.join(HERE, "ppzj-moments_NNLO.xml")
FO2_LABEL = r"fixed order (NNLO $Z$+jet)"
LAB_NLO = r"MaxEnt, NLO$(Zj)$ recoil"
LAB_NNLO = r"MaxEnt, NNLO$(Zj)$ recoil"


def dens(x, w, e):
    h, _ = np.histogram(x, e, weights=w / w.sum()); return h / np.diff(e)


def main():
    check_seam(XM, Q_HARD, label="Drell-Yan")
    seeds = common_seeds(BASE, "DY_MOMENTS", CH6)
    print(f"DY FO seeds usable: {seeds}")
    M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, seeds,
                                       born_tags={"mll": "mll", "y_abs": "absyz"},
                                       n_born=12, n_recoil=20,
                                       x_match=XM, x_hi=XMAP, soft_lo=SOFT)
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
               recoil={"pT_ll": {"range": (SOFT, XMAP), "map": "log", "soft_lo": SOFT,
                                 "profile": {"a": XM, "b": 2 * XM, "c": XHI}}},
               followers=["phistar"])
    print("solving ...", flush=True)
    res = upgrade(ev, M, cfg)
    print(f"  effN {100*res.effN:.1f}%  closure {res.closure:.2e}  "
          f"neg-wt {100*np.mean(res.weights<=0):.1f}%")
    res2 = None
    if os.path.exists(NNLO_XML):
        os.environ["DY_RECOIL_XML"] = NNLO_XML
        try:
            M2 = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, seeds,
                                                born_tags={"mll": "mll", "y_abs": "absyz"},
                                                n_born=12, n_recoil=20,
                                                x_match=XM, x_hi=XMAP, soft_lo=SOFT)
        finally:
            os.environ.pop("DY_RECOIL_XML", None)
        src = M2["recoil"]["pT_ll"]["_source"]
        print(f"  NNLO(Zj) recoil from {os.path.basename(src['file'])}: {src['n_orders']} orders, "
              f"W0 {src['W0_pb']:.2f} pb / sigma_fid {src['sigma_fid_pb']:.2f} pb -> R {M2['recoil']['pT_ll']['rate']:.4f}")
        res2 = upgrade(ev, M2, cfg)
        print(f"  NNLO(Zj) recoil: effN {100*res2.effN:.1f}%  closure {res2.closure:.2e}  "
              f"neg-wt {100*np.mean(res2.weights<=0):.1f}%  orders {res2.report['moment_selection']['chosen']}")

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
    mx2_scale_w = mx2_boot_w = None
    bwf2 = os.path.join(HERE, "dy_band_weights_nnloZj.npz")
    if res2 is not None and os.path.exists(bwf2):
        BW2 = dict(np.load(bwf2))
        if np.allclose(BW2["central"], res2.weights, rtol=1e-6, atol=0):
            mx2_scale_w, mx2_boot_w = BW2["scale_w"], BW2["boot_w"]
        else:
            print("  WARNING: dy_band_weights_nnloZj.npz stale; no bands on the NNLO(Zj) curve")

    # Two ratio panels to the data: (r) the upgrade against its prior and the
    # fixed-order input; (r2) the upgrade against the matched generators.
    fig, ax = plt.subplots(3, len(PANELS), figsize=(6.9 * len(PANELS), 10.6), squeeze=False,
                           gridspec_kw={"height_ratios": [2.2, 1.0, 1.0], "hspace": 0.07,
                                        "wspace": 0.24})
    summary = {}
    summaryA = {}
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
        # Above-the-seam split: the papers quote pT_ll agreement separately for
        # the region the matching actually governs.  Same metric and same upper
        # cut as med(), restricted to bins whose LOW edge is at or above the
        # seam, i.e. the split at the dashed line the figure already draws.
        mA = m & (lo >= XM) if key == "pT_ll" else m
        medA = lambda h: 100 * np.median(np.abs(h[mA] / val[mA] - 1)) if mA.any() else float("nan")

        a.errorbar(ctr[m], val[m], yerr=err[m], fmt="o", color="k", ms=4, lw=1.1,
                   label=r"ATLAS 1912.02844", zorder=10)
        hp = dens(ev[key], ev["weight"], e)
        hq = dens(ev[key], res.weights, e)
        a.stairs(np.where(m, hp, np.nan), e, color=C["prior"], ls="--", lw=2.0,
                 label=rf"PS+LO prior ({med(hp):.1f}\%)")
        a.stairs(np.where(m, hq, np.nan), e, color=C["maxent"], lw=3.2,
                 label=rf"{LAB_NLO} ({med(hq):.1f}\%, $0\%\,w<0$)")
        hq2 = dens(ev[key], res2.weights, e) if res2 is not None else None
        if hq2 is not None:
            a.stairs(np.where(m, hq2, np.nan), e, color=C["maxent_nnlo"], lw=3.2,
                     label=rf"{LAB_NNLO} ({med(hq2):.1f}\%, $0\%\,w<0$)")
        r.stairs(np.where(m, hp / np.maximum(val, 1e-30), np.nan), e, color=C["prior"], ls="--", lw=1.8)
        for rr_ in (r, r2):
            rr_.stairs(np.where(m, hq / np.maximum(val, 1e-30), np.nan), e, color=C["maxent"], lw=2.6)
            if hq2 is not None:
                rr_.stairs(np.where(m, hq2 / np.maximum(val, 1e-30), np.nan), e, color=C["maxent_nnlo"], lw=2.6)
        summary[key] = {"prior": med(hp), "MaxEnt": med(hq)}
        summaryA[key] = {"prior": medA(hp), "MaxEnt": medA(hq)}
        if hq2 is not None:
            summary[key]["MaxEnt NNLO(Zj) recoil"] = med(hq2)
            summaryA[key]["MaxEnt NNLO(Zj) recoil"] = medA(hq2)
        vref = np.maximum(val, 1e-30)
        NSER = 6   # prior, MaxEnt x2, 3 generators -- for bar staggering

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
        if hq2 is not None and mx2_scale_w is not None:
            hs2 = np.array([dens(ev[key], w_, e) for w_ in mx2_scale_w])
            lo2, hi2 = np.minimum(hq2, hs2.min(0)), np.maximum(hq2, hs2.max(0))
            for rr_ in (r, r2):
                rr_.fill_between(ctr, np.where(m, lo2 / vref, np.nan),
                                 np.where(m, hi2 / vref, np.nan),
                                 step="mid", color=C["maxent_nnlo"], alpha=0.15, lw=0)
            boot2 = np.array([dens(ev[key], w_, e)
                              for w_ in mx2_boot_w]).std(0, ddof=1)
            stat_bars(ev[key], res2.weights, hq2, C["maxent_nnlo"], 2, extra=boot2, axes=(r, r2))
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
                WS = gen_scale_weights(G)
                hgs = np.array([dens(gx, WS[:, kk], e) for kk in range(WS.shape[1])])
                r2.fill_between(ctr, np.where(m, np.minimum(hg, hgs.min(0)) / vref, np.nan),
                                np.where(m, np.maximum(hg, hgs.max(0)) / vref, np.nan),
                                step="mid", color=col, alpha=0.12, lw=0)
            stat_bars(gx, gw, hg, col, 3 + kg, axes=(r2,))
            summary[key][lbl] = med(hg)
            summaryA[key][lbl] = medA(hg)
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
            # Fixed order (NLO Z+jet, the real channels of the NNLO Z run)
            # normalised ABSOLUTELY to the NNLO fiducial cross section and
            # drawn everywhere, below the seam included (ptz_full, separate
            # production) -- no window anchor, no boundary.  Above the seam
            # the absolute normalisation equals the profile-rate anchor by
            # construction (the n=0 recoil moment is the same on both sides).
            from fo_ptll_full import fo_ptll
            fn, fhalf, fst, have_full, sig0 = fo_ptll(e, seam=XM)
            inw = (ev[key] >= XM) & (ev[key] < XHI)
            tgt = float(res.weights[inw].sum() / res.weights.sum())
            print(f"  FO pT_ll: sigma_fid(NNLO)={sig0:.6g}; below-seam histogram "
                  f"{'present' if have_full else 'ABSENT -- drawn above the seam only'}; "
                  f"FO sharp-window integral {np.nansum(np.where(ctr >= XM, fn, 0) * np.diff(e)):.4f} "
                  f"vs MaxEnt sharp-window rate {tgt:.4f}")
            if np.isfinite(fn).any():
                a.stairs(np.where(fn > 0, fn, np.nan), e, color=C["fo"], ls=":", lw=LW["fo"],
                         label=FO_LABEL)
                rr_ = fn / vref
                rb_ = fhalf / vref                        # BAND = scale only
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
        if key == "pT_ll" and res2 is not None:
            # NNLO Z+jet fixed order (Stripper): the direct pT_Z histogram on the
            # data edges, normalised to OUR NNLO fiducial cross section at the
            # paired scale choice (the convention of the NLO curve and of the
            # moment hook), band = his 7-point envelope, bars = his own errors.
            from stripper_moments import load_stripper_hist
            from fo_ptll_full import sigma_fid
            H = load_stripper_hist(NNLO_XML, "pT_Z", sigma_fid() / 1000.0)
            if not np.allclose(H["edges"], e):
                raise ValueError("Stripper pT_Z edges differ from the ATLAS edges")
            hn = np.where(e[:-1] >= XM - 1e-9, H["dens"], np.nan)
            hn_lo = np.minimum(hn, np.nanmin(H["members"], 0)); hn_hi = np.maximum(hn, np.nanmax(H["members"], 0))
            a.stairs(np.where(hn > 0, hn, np.nan), e, color=C["fo"], ls="-.", lw=LW["fo"], label=FO2_LABEL)
            for ax_ in (r,):      # the input panel only; r2 is the generator comparison
                ax_.errorbar(stagger(e, 5, NSER), np.where(m, hn / vref, np.nan),
                             yerr=np.where(m, H["stat"] / vref, np.nan), fmt="none",
                             ecolor=C["fo"], elinewidth=1.0, capsize=1.6, alpha=0.8)
                ax_.stairs(np.where(m, hn / vref, np.nan), e, color=C["fo"], ls="-.", lw=2.0)
                ax_.fill_between(ctr, np.where(m, hn_lo / vref, np.nan), np.where(m, hn_hi / vref, np.nan),
                                 step="mid", color=C["fo"], alpha=0.15, lw=0)
            relerr = H["stat"] / np.abs(H["dens"])
            print(f"  FO NNLO(Zj) pT_ll (Stripper): integral above the seam "
                  f"{np.nansum(np.where(e[:-1] >= XM, hn, 0) * np.diff(e)):.4f} of sigma_fid; "
                  f"median |ratio-1| to data above the seam {100*np.nanmedian(np.abs(hn[mA]/val[mA]-1)):.1f}%; "
                  f"per-bin errors 30-85 GeV {100*np.min(relerr[(lo>=30)&(hi<=85)]):.0f}-{100*np.max(relerr[(lo>=30)&(hi<=85)]):.0f}%, "
                  f"above 200 GeV {100*np.max(relerr[(lo>=200)&(hi<=400)]):.0f}% at most")
        rel = err / np.maximum(val, 1e-30)
        for rr_ in (r, r2):
            rr_.fill_between(ctr[m], (1 - rel)[m], (1 + rel)[m], color=C["band"], alpha=0.55, step="mid")
            rr_.axhline(1, color="k", lw=0.8)
        if key == "pT_ll":
            # the matching scale itself (exact on this axis); no window shading
            for p_ in (a, r, r2):
                p_.axvline(XM, color=C["seam"], lw=2.0, ls="--")
        a.set_xscale("log"); a.set_yscale("log"); r.set_xscale("log"); r2.set_xscale("log")
        # top panel: the data's own range, no empty decades
        a.set_ylim(0.3 * np.nanmin(val[m]), 2.5 * np.nanmax(val[m]))
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
    # James's suggestion (tail-only top panel) as a side-by-side variant, NOT the paper figure
    ax[0, 0].set_ylim(top=1e-3); fig.savefig(out.replace(".pdf", "_crop.png"))
    for k, v in summary.items():
        print(f"  {k}: " + "  ".join(f"{a_}={b_:.2f}%" for a_, b_ in v.items()))
    for k, v in summaryA.items():
        if k != "pT_ll":
            continue
        print(f"  {k} ABOVE SEAM (pT>{XM:g} GeV): "
              + "  ".join(f"{a_}={b_:.2f}%" for a_, b_ in v.items()))


if __name__ == "__main__":
    main()
