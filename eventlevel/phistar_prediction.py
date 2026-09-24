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
from pubstyle import use_pub_style, C, LW, gen_scale_weights
use_pub_style()
from maxent_upgrade import upgrade, check_seam
from nnlojet_moments import (fo_moments_smooth_from_nnlojet, common_seeds,
                             fo_curve_band, fo_curve)
BASE = os.path.join(os.environ.get("NNLOJET_ROOT",
        os.path.expanduser("~/nnlojet-v1.0.2")), "dy_profile_log30_hi")
# mirrors eval_w_ptz (pa=30, pb=60, pc=pd=500)
XM, XHI, SOFT, XMAP = 30.0, 500.0, 30.0, 2500.0
Q_HARD = 91.1876
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
# phi* > 0 REQUIRES a real emission, so from the inclusive Z calculation at
# NNLO it is only NLO(Z+jet) accurate -- the Born observables m_ll and |y_ll|
# are the NNLO ones.  Label the curve by the accuracy of THIS observable.
FO_LABEL = r"fixed order (NLO $Z$+jet)"
SEEDS = None     # resolved at run time to every channel-complete seed
# Second upgrade: the same Born towers, the recoil from the NNLO Z+jet Stripper
# calculation (DY_RECOIL_XML hook); phi* stays a pure follower in both.
NNLO_XML = os.path.join(HERE, "ppzj-moments_NNLO.xml")
FO2_LABEL = r"fixed order (NNLO $Z$+jet)"
# NOT DRAWN (2026-09-23): his phietastar histogram carries only 3.5% of sigma_fid
# above phi* = 0.391 against 10.6% in the data, 8.7% in the prior and 8.5% in our
# NLO(Zj) fixed order, although 99.6% of those events have pT_ll > 30 GeV and his
# pT_Z spectrum is complete there -- his phi* is not the ATLAS phi*_eta (definition
# to be settled with R. Poncelet).  Set a threshold to draw it once it is.
FO2_PHISTAR_MIN = None
LAB_NLO = r"MaxEnt, NLO$(Zj)$ recoil"
LAB_NNLO = r"MaxEnt, NNLO$(Zj)$ recoil"


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
                                       n_born=12, n_recoil=20, x_match=XM, x_hi=XMAP, soft_lo=SOFT)
    print("solving ...", flush=True)
    res = upgrade(ev, M, cfg())
    print(f"  effN={100*res.effN:.0f}%  closure={res.closure:.1e}  neg-wt={100*np.mean(res.weights<=0):.0f}%", flush=True)
    res2 = None
    if os.path.exists(NNLO_XML):
        os.environ["DY_RECOIL_XML"] = NNLO_XML
        try:
            M2 = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, (SEEDS or common_seeds(BASE, 'DY_MOMENTS', CH6)),
                                                born_tags={"mll": "mll", "y_abs": "absyz"},
                                                n_born=12, n_recoil=20, x_match=XM, x_hi=XMAP, soft_lo=SOFT)
        finally:
            os.environ.pop("DY_RECOIL_XML", None)
        res2 = upgrade(ev, M2, cfg())
        print(f"  NNLO(Zj) recoil ({M2['recoil']['pT_ll']['_source']['n_orders']} orders, R {M2['recoil']['pT_ll']['rate']:.4f}): "
              f"effN={100*res2.effN:.1f}%  closure={res2.closure:.1e}  neg-wt={100*np.mean(res2.weights<=0):.0f}%", flush=True)

    D = dict(np.load(os.path.join(HERE, "atlas_phistar_born.npz")))
    e = np.concatenate([D["lo"][:1], D["hi"]]); ctr = D["center"]
    val, err = D["val"], D["err"]; msk = val > 0

    # fixed order: channel-summed, seed-pooled phi* reference, rebinned onto
    # the DATA edges per seed (exact overlap integration), with the full
    # uncertainty decomposition: 7-point scale envelope (+) seed scatter.
    fo_h = fo_band = None
    fb = fo_curve_band(BASE, "DY_MOMENTS", CH6,
                       common_seeds(BASE, "DY_MOMENTS", CH6, tag="phistar_a"),
                       "phistar_a", edges=e, members=True)
    if fb is not None:
        _, _, fo_h, _, _, fst, fo_mem = fb
        # ABSOLUTE normalisation, no boundary choice: (1/sigma_NNLO) dsigma/dphi*
        # with sigma_NNLO the fiducial cross section of the same run at the
        # same scale (mll_fine integrated over the mass window, all six
        # channels; every fiducial event has mll in the window).  phi* > 0
        # receives only the real channels, so this is the NNLO-normalised
        # spectrum at NLO(Z+jet) accuracy; below phi* ~ 0.01 it is negative and
        # between 0.01 and 0.3 it overshoots the data, its logarithms being
        # unresummed -- drawn as is, at full strength, with no seam marker.
        sig = []
        for s_ in range(7):
            lo_m, hi_m, d_m, _ = fo_curve(BASE, "DY_MOMENTS", CH6,
                                          common_seeds(BASE, "DY_MOMENTS", CH6, tag="mll_fine"),
                                          "mll_fine", scale_idx=s_)
            inr = (lo_m >= 66 - 1e-9) & (hi_m <= 116 + 1e-9)
            sig.append(float(np.sum(d_m[inr] * (hi_m - lo_m)[inr])))
        fo_h = fo_h / sig[0]; fst = fst / sig[0]
        mem_n = np.array([fo_mem[s_] / sig[s_] for s_ in range(7)])
        fsl = np.minimum(fo_h, np.nanmin(mem_n, 0)); fsh = np.maximum(fo_h, np.nanmax(mem_n, 0))
        fo_scale = 0.5 * (fsh - fsl)   # BAND
        fo_stat = fst                  # CANDLE
        fo_band = fo_scale             # (kept: band is scale only)
        print(f"  FO phi*: sigma_fid(NNLO) = {sig[0]:.6g}, integral of the phi*>0.004 spectrum = "
              f"{np.nansum(fo_h * np.diff(e)):.4f} of sigma_fid")
    fo2_h = None
    if fb is not None and res2 is not None and FO2_PHISTAR_MIN is not None:
        from stripper_moments import load_stripper_hist
        H2 = load_stripper_hist(NNLO_XML, "phietastar", np.array(sig) / 1000.0)
        if not np.allclose(H2["edges"], e):
            raise ValueError("Stripper phi* edges differ from the ATLAS edges")
        keep = e[:-1] >= FO2_PHISTAR_MIN - 1e-9
        fo2_h = np.where(keep, H2["dens"], np.nan)
        fo2_lo = np.minimum(fo2_h, np.nanmin(H2["members"], 0)); fo2_hi = np.maximum(fo2_h, np.nanmax(H2["members"], 0))
        fo2_stat = np.where(keep, H2["stat"], np.nan)
        mk = keep & msk
        print(f"  FO NNLO(Zj) phi* (Stripper) above {FO2_PHISTAR_MIN}: median |ratio-1| to data "
              f"{100*np.nanmedian(np.abs(fo2_h[mk]/val[mk]-1)):.1f}%, per-bin errors "
              f"{100*np.nanmin(fo2_stat[mk]/fo2_h[mk]):.0f}-{100*np.nanmax(fo2_stat[mk]/fo2_h[mk]):.0f}%")

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
    mx_ws2 = mx_boot2 = None
    bwf2 = os.path.join(HERE, "dy_band_weights_nnloZj.npz")
    if res2 is not None and os.path.exists(bwf2):
        BW2 = dict(np.load(bwf2))
        if np.allclose(BW2["central"], res2.weights, rtol=1e-6, atol=0):
            mx_ws2 = np.column_stack([res2.weights] + list(BW2["scale_w"]))
            mx_boot2 = np.array([norm_dens(ev["phistar"], w, e)
                                 for w in BW2["boot_w"]]).std(0, ddof=1)
        else:
            print("  WARNING: dy_band_weights_nnloZj.npz stale; no bands on the NNLO(Zj) curve")

    # (label, x, w, w_scale|None, boot|None, color, ls, neg%)
    series = [(r"PS+LO prior", ev["phistar"], ev["weight"], None, None,
               "0.55", "--", None),
              (LAB_NLO, ev["phistar"], res.weights, mx_ws, mx_boot,
               C["maxent"], "-", 0)]
    if res2 is not None:
        series.append((LAB_NNLO, ev["phistar"], res2.weights, mx_ws2, mx_boot2,
                       C["maxent_nnlo"], "-", 0))
    for lbl, f, col, neg in [(r"MiNNLO$_{\mathrm{PS}}$", "dy_minnlo_atlas_v4.npz", C["minnlo"], 23),
                             (r"MC@NLO", "dy_mcatnlo_atlas_v4.npz", C["mcatnlo"], 5),
                             (r"POWHEG", "dy_powheg_atlas_v6.npz", C["powheg"], 1)]:
        G = dict(np.load(os.path.join(HERE, f)))
        series.append((lbl, G["phistar"].astype(float), G["w"].astype(float),
                       (gen_scale_weights(G) if "w_scale" in G else None),
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
        # Drawn everywhere at full strength: where fixed order fails (small
        # phi*, unresummed logarithms) the curve simply leaves the data, and
        # the reader sees it, without any boundary being drawn.
        above = np.ones(len(ctr), bool)
        for p_, h_ in ((a, fo_h), (r, fo_h / np.maximum(val, 1e-30)), (r2, fo_h / np.maximum(val, 1e-30))):
            p_.stairs(np.where(msk, h_, np.nan), e, color="k", ls=":",
                      lw=LW["fo"], label=(FO_LABEL if p_ is a else None))
        # its own scale (+) stat band, same convention as every other curve
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
    if fo2_h is not None:
        rr2 = fo2_h / np.maximum(val, 1e-30)
        a.stairs(np.where(msk, fo2_h, np.nan), e, color="k", ls="-.", lw=LW["fo"], label=FO2_LABEL)
        for rr_ in (r, r2):
            rr_.stairs(np.where(msk, rr2, np.nan), e, color="k", ls="-.", lw=LW["fo"])
            rr_.fill_between(ctr, np.where(msk, fo2_lo / np.maximum(val, 1e-30), np.nan),
                             np.where(msk, fo2_hi / np.maximum(val, 1e-30), np.nan),
                             step="mid", color="k", alpha=0.15, lw=0)
            rr_.errorbar(ctr, np.where(msk, rr2, np.nan),
                         yerr=np.where(msk, fo2_stat / np.maximum(val, 1e-30), np.nan),
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
    # NO seam marker on phi*: phi* ~ a_T/m with a_T <= pT_ll, so the matching
    # scale has no sharp image here (dropped 2026-09-06).
    a.set_xscale("log"); a.set_yscale("log"); r.set_xscale("log"); r2.set_xscale("log")
    # top panel: the data's own range, no empty decades
    a.set_ylim(0.3 * np.nanmin(val[msk]), 2.5 * np.nanmax(val[msk]))
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
    # James's suggestion (tail-only top panel) as a side-by-side variant, NOT the paper figure
    a.set_ylim(top=1e-2); fig.savefig(out.replace(".pdf", "_crop.png"))
    for lbl, x, w, *_ in series:
        print(f"  {lbl:14s} med|ratio-1| = {med(norm_dens(x, w, e)):5.2f}%")


def cfg():
    # "bw" MUST match the Breit-Wigner map now compiled into eval_chebT_mll --
    # the moment files were regenerated with it; "lin" here silently mismatches.
    return dict(born={"mll": {"range": (66., 116.), "map": "bw"}, "y_abs": {"range": (0., 2.4), "map": "lin"}},
                recoil={"pT_ll": {"range": (SOFT, XMAP), "map": "log", "soft_lo": SOFT,
                                  "profile": {"a": XM, "b": 2 * XM, "c": XHI}}},
                followers=["phistar"])


if __name__ == "__main__":
    main()
