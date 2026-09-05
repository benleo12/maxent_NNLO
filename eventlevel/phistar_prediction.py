#!/usr/bin/env python3
r"""PRL money plot: phi*_eta is NEVER constrained (a pure follower), yet the
MaxEnt-reweighted PS+LO sample predicts it as well as / better than the matched
generators (MiNNLO$_{\\mathrm{PS}}$, MC@NLO, POWHEG) -- with 0% negative weights.

We impose FO moments ONLY on {m_ll, |y_ll|, pT_ll}.  phi* is determined by the
kinematics we did NOT constrain; that it comes out right is the point.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import use_pub_style, C, LW
use_pub_style()
from maxent_upgrade import upgrade, check_seam
from nnlojet_moments import (fo_moments_smooth_from_nnlojet, common_seeds,
                             fo_curve_band)
BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_poc"
# mirrors eval_w_ptz (pa=30, pb=60, pc=pd=500)
XM, XHI, SOFT = 30.0, 500.0, 0.5
Q_HARD = 91.1876
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
# phi* > 0 REQUIRES a real emission, so from the inclusive Z calculation at
# NNLO it is only NLO(Z+jet) accurate -- the Born observables m_ll and |y_ll|
# are the NNLO ones.  Label the curve by the accuracy of THIS observable.
FO_LABEL = r"fixed order (NLO $Z$+jet)"
SEEDS = None     # resolved at run time to every channel-complete seed


def dens(x, w, e):
    bw = np.diff(e); h, _ = np.histogram(x, e, weights=w / w.sum()); return h / bw


def main():
    check_seam(XM, Q_HARD, label="Drell-Yan")
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v3.npz")))
    n = len(P["w"]); idx = np.random.default_rng(0).choice(n, min(1_000_000, n), replace=False)
    ev = dict(mll=P["mll"][idx].astype(float), y_abs=np.abs(P["y_ll"][idx]).astype(float),
              pT_ll=P["pT_ll"][idx].astype(float), phistar=P["phistar"][idx].astype(float),
              weight=P["w"][idx].astype(float))
    M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, (SEEDS or common_seeds(BASE, 'DY_MOMENTS', CH6)),
                                       born_tags={"mll": "mll", "y_abs": "absyz"},
                                       n_born=6, n_recoil=12, x_match=XM, x_hi=XHI, soft_lo=SOFT)
    print("solving ...", flush=True)
    res = upgrade(ev, M, cfg())
    print(f"  effN={100*res.effN:.0f}%  closure={res.closure:.1e}  neg-wt={100*np.mean(res.weights<=0):.0f}%", flush=True)

    D = dict(np.load(os.path.join(HERE, "atlas_phistar_born.npz")))
    e = np.concatenate([D["lo"][:1], D["hi"]]); ctr = D["center"]
    val, err = D["val"], D["err"]; msk = val > 0

    # fixed order: channel-summed, seed-pooled phi* reference, rebinned onto
    # the DATA edges per seed (exact overlap integration), with the full
    # uncertainty decomposition: 7-point scale envelope (+) seed scatter.
    fo_h = fo_band = None
    fb = fo_curve_band(BASE, "DY_MOMENTS", CH6,
                       common_seeds(BASE, "DY_MOMENTS", CH6, tag="phistar_a"),
                       "phistar_a", edges=e)
    if fb is not None:
        _, _, fo_h, fsl, fsh, fst = fb
        # Normalise over the region where fixed order is TRUSTED, i.e. above
        # the seam.  Normalising over the full range lets the unresummed
        # Sudakov region below the seam -- where the fixed order diverges
        # well above the data -- eat the normalisation and push the trusted
        # part down by 17% (median FO/data 0.83 instead of 1.03).  The SAME
        # factor multiplies every scale variant, so the band keeps the FO
        # curve's own relative scale variation.
        xs_ = XM / 91.1876
        bwe = np.diff(e); ab_ = ctr >= xs_
        fo_above = np.nansum(np.where(ab_, fo_h, 0.0) * bwe)
        d_above = float((val * bwe * ab_).sum()); d_tot = float((val * bwe).sum())
        if fo_above > 0:
            k_ = (d_above / d_tot) / fo_above
            fo_h, fsl, fsh, fst = fo_h * k_, fsl * k_, fsh * k_, fst * k_
        fo_scale = 0.5 * (fsh - fsl)   # BAND
        fo_stat = fst                  # CANDLE
        fo_band = fo_scale             # (kept: band is scale only)

    # uncertainty inputs: MaxEnt scale/bootstrap variants from moment_bands,
    # generator scale weights from the regenerated npz (w_scale, 7-point)
    from bandviz import norm_dens, draw_series_unc
    mx_ws = mx_boot = None
    bwf = os.path.join(HERE, "dy_band_weights.npz")
    if os.path.exists(bwf):
        BW = dict(np.load(bwf))
        if np.allclose(BW["central"], res.weights, rtol=1e-6, atol=0):
            mx_ws = np.column_stack([res.weights] + list(BW["scale_w"]))
            mx_boot = np.array([norm_dens(ev["phistar"], w, e)
                                for w in BW["boot_w"]]).std(0, ddof=1)
        else:
            print("  WARNING: dy_band_weights.npz stale; no MaxEnt bands")

    # (label, x, w, w_scale|None, boot|None, color, ls, neg%)
    series = [(r"PS+LO prior", ev["phistar"], ev["weight"], None, None,
               "0.55", "--", None),
              (r"MaxEnt", ev["phistar"], res.weights, mx_ws, mx_boot,
               C["maxent"], "-", 0)]
    for lbl, f, col, neg in [(r"MiNNLO$_{\mathrm{PS}}$", "dy_minnlo_atlas_v3.npz", C["minnlo"], 23),
                             (r"MC@NLO", "dy_mcatnlo_atlas_v4.npz", C["mcatnlo"], 5),
                             (r"POWHEG", "dy_powheg_atlas_v4.npz", C["powheg"], 1)]:
        G = dict(np.load(os.path.join(HERE, f)))
        series.append((lbl, G["phistar"].astype(float), G["w"].astype(float),
                       (G["w_scale"].astype(float) if "w_scale" in G else None),
                       None, col, "-", neg))

    def med(h):
        m = msk & (h > 0); return 100 * np.median(np.abs(h[m] / val[m] - 1))

    # Two ratio panels to the data: (r) the upgrade against its prior and the
    # fixed order; (r2) the upgrade against the matched generators.
    fig, ax = plt.subplots(3, 1, figsize=(8.6, 11.2),
                           gridspec_kw={"height_ratios": [2.3, 1.0, 1.0], "hspace": 0.06})
    a, r, r2 = ax
    rel = err / np.maximum(val, 1e-30)
    a.errorbar(ctr[msk], val[msk], yerr=err[msk], fmt="o", color="k", ms=5, lw=1.2,
               label=r"ATLAS 1912.02844", zorder=10)
    if fo_h is not None:
        # Fixed order is a prediction only above the seam.  Below it the
        # Sudakov logarithms are unresummed and the curve is meaningless --
        # draw it faded there rather than letting its spikes dominate the eye.
        xs_ = XM / 91.1876
        above = ctr >= xs_
        for p_, h_ in ((a, fo_h), (r, fo_h / np.maximum(val, 1e-30)), (r2, fo_h / np.maximum(val, 1e-30))):
            p_.stairs(np.where(msk & above, h_, np.nan), e, color="k", ls=":",
                      lw=LW["fo"], label=(FO_LABEL if p_ is a else None))
            p_.stairs(np.where(msk & ~above, h_, np.nan), e, color="k", ls=":",
                      lw=1.4, alpha=0.22)
        # its own scale (+) stat band, same convention as every other curve;
        # only above the seam, where the curve means something
        rrat = fo_h / np.maximum(val, 1e-30)
        rband = fo_scale / np.maximum(val, 1e-30)
        for rr_ in (r, r2):
            rr_.fill_between(ctr, np.where(msk & above, rrat - rband, np.nan),
                             np.where(msk & above, rrat + rband, np.nan),
                             step="mid", color="k", alpha=0.15, lw=0)
            # CANDLE = the fixed order's own seed scatter, the same split every
            # other curve on this panel uses (band = scale, bars = statistics)
            rr_.errorbar(ctr, np.where(msk & above, rrat, np.nan),
                         yerr=np.where(msk & above, fo_stat / np.maximum(val, 1e-30), np.nan),
                         fmt="none", ecolor="k", elinewidth=1.0, capsize=1.6, alpha=0.75)
    nser = len(series)
    for k, (lbl, x, w, ws, boot, col, ls, neg) in enumerate(series):
        h = norm_dens(x, w, e)
        lw = 3.2 if "MaxEnt" in lbl else 1.9
        leg = (rf"{lbl} ({med(h):.1f}\%)" if neg is None
               else rf"{lbl} ({med(h):.1f}\%, {neg}\% $w<0$)")
        a.stairs(np.where(msk, h, np.nan), e, color=col, ls=ls, lw=lw, label=leg)
        # prior -> upper ratio; generators -> lower ratio; the upgrade on both
        targets = (r, r2) if "MaxEnt" in lbl else ((r,) if "prior" in lbl else (r2,))
        for rr_ in targets:
            rr_.stairs(np.where(msk, h / np.maximum(val, 1e-30), np.nan), e, color=col, ls=ls, lw=lw)
            # scale band (translucent, this sample's 7-point) + staggered stat bars
            draw_series_unc(rr_, x, w, ws, e, np.where(msk, val, np.nan), col, k, nser,
                            boot=boot, alpha=0.13)
    for rr_ in (r, r2):
        rr_.fill_between(ctr[msk], (1 - rel)[msk], (1 + rel)[msk], color=C["band"], alpha=0.5, step="mid")
        rr_.axhline(1, color="k", lw=0.8)
    # the pT seam maps onto phi* via phi* ~ pT/m  =>  mark it
    xs = XM / 91.1876
    for p_ in (a, r, r2):
        p_.axvline(xs, color=C["seam"], lw=1.8, ls="--")
    a.set_xscale("log"); a.set_yscale("log"); r.set_xscale("log"); r2.set_xscale("log")
    # axes-fraction coords: immune to the y-scale being set after this call.
    # (Reading get_ylim() on the still-linear axis put the label at a negative y,
    # which is off-scale once the axis goes log -- bbox="tight" then grew the
    # canvas to 56000 pt to contain it.)
    a.tick_params(labelbottom=False); r.tick_params(labelbottom=False)
    r.set_ylim(0.80, 1.20); r2.set_ylim(0.80, 1.20)
    a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}\phi^*_\eta$")
    r.set_ylabel(r"ratio to data"); r2.set_ylabel(r"ratio to data"); r2.set_xlabel(r"$\phi^*_\eta$")
    r.text(0.02, 0.92, "upgrade vs prior and fixed order", transform=r.transAxes, fontsize=11, va="top")
    r2.text(0.02, 0.92, "upgrade vs matched generators", transform=r2.transAxes, fontsize=11, va="top")
    x0 = e[e > 0].min(); a.set_xlim(x0, e[-1]); r.set_xlim(x0, e[-1]); r2.set_xlim(x0, e[-1])
    a.legend(loc="lower left", bbox_to_anchor=(0.015, 0.015), handlelength=1.4,
             labelspacing=0.28, borderaxespad=0.0)
    a.set_title(r"$\phi^*_\eta$, never constrained")
    out = os.path.join(HERE, "fig_phistar_prediction.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    print("wrote", out)
    for lbl, x, w, *_ in series:
        print(f"  {lbl:14s} med|ratio-1| = {med(norm_dens(x, w, e)):5.2f}%")


def cfg():
    # "bw" MUST match the Breit-Wigner map now compiled into eval_chebT_mll --
    # the moment files were regenerated with it; "lin" here silently mismatches.
    return dict(born={"mll": {"range": (66., 116.), "map": "bw"}, "y_abs": {"range": (0., 2.4), "map": "lin"}},
                recoil={"pT_ll": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                                  "profile": {"a": XM, "b": 2 * XM, "c": XHI}}},
                followers=["phistar"])


if __name__ == "__main__":
    main()
