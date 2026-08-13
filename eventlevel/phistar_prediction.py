#!/usr/bin/env python3
r"""PRL money plot: phi*_eta is NEVER constrained (a pure follower), yet the
MaxEnt-reweighted PS+LO sample predicts it as well as / better than the matched
generators (MiNNLO, MC@NLO, POWHEG) -- with 0% negative weights.

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
from pubstyle import use_pub_style, C, LW, rebin_density
use_pub_style()
from maxent_upgrade import upgrade, check_seam
from nnlojet_moments import fo_moments_smooth_from_nnlojet, common_seeds, _load
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
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v2.npz")))
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

    # fixed order: channel-summed, seed-pooled phi* reference
    def fo_curve(tag):
        lo = hi = None; tot = None
        for s_ in common_seeds(BASE, "DY_MOMENTS", CH6, tag=tag):
            for ch in CH6:
                r0 = _load(os.path.join(BASE, f"ch_{ch}", f"Z.DY_MOMENTS.{ch}.{tag}.s{s_}.dat"))
                if r0 is None: continue
                lo, _, hi, v, _ = r0
                tot = v[:, 0].copy() if tot is None else tot + v[:, 0]
        return (lo, hi, tot) if tot is not None else None

    fo_h = None
    fc = fo_curve("phistar_a")
    if fc is not None:
        flo, fhi, fv = fc
        gd = (fv > 0) & (fhi > flo)
        if gd.sum() > 2:
            # SAME EDGES as data / prior / MaxEnt / generators, by exact overlap
            # integration.  Assigning native bins to data bins by their centre
            # (what this used to do) drops or double-counts bins whenever the
            # two grids are not nested, and does not conserve the integral.
            fo_h = rebin_density(flo[gd], fhi[gd], fv[gd], e)
            # Normalise over the region where fixed order is TRUSTED, i.e. above
            # the seam.  Normalising over the full range lets the unresummed
            # Sudakov region below the seam -- where the fixed order diverges
            # well above the data -- eat the normalisation and push the trusted
            # part down by 17% (median FO/data 0.83 instead of 1.03).
            xs_ = XM / 91.1876
            bwe = np.diff(e); ab_ = ctr >= xs_
            fo_above = np.nansum(np.where(ab_, fo_h, 0.0) * bwe)
            d_above = float((val * bwe * ab_).sum()); d_tot = float((val * bwe).sum())
            if fo_above > 0:
                fo_h = fo_h * (d_above / d_tot) / fo_above

    series = [(r"LO+PS prior", dens(ev["phistar"], ev["weight"], e), "0.55", "--", None),
              (r"MaxEnt", dens(ev["phistar"], res.weights, e), C["maxent"], "-", 0)]
    for lbl, f, col, neg in [(r"MiNNLO", "dy_minnlo_atlas_v2.npz", C["minnlo"], 23),
                             (r"MC@NLO", "dy_mcatnlo_atlas.npz", C["mcatnlo"], 5),
                             (r"POWHEG", "dy_powheg_atlas.npz", C["powheg"], 1)]:
        G = dict(np.load(os.path.join(HERE, f)))
        series.append((lbl, dens(G["phistar"].astype(float), G["w"].astype(float), e), col, "-", neg))

    def med(h):
        m = msk & (h > 0); return 100 * np.median(np.abs(h[m] / val[m] - 1))

    fig, ax = plt.subplots(2, 1, figsize=(8.6, 8.8),
                           gridspec_kw={"height_ratios": [2.3, 1.1], "hspace": 0.06})
    a, r = ax
    rel = err / np.maximum(val, 1e-30)
    a.errorbar(ctr[msk], val[msk], yerr=err[msk], fmt="o", color="k", ms=5, lw=1.2,
               label=r"ATLAS 1912.02844", zorder=10)
    if fo_h is not None:
        # Fixed order is a prediction only above the seam.  Below it the
        # Sudakov logarithms are unresummed and the curve is meaningless --
        # draw it faded there rather than letting its spikes dominate the eye.
        xs_ = XM / 91.1876
        above = ctr >= xs_
        for p_, h_ in ((a, fo_h), (r, fo_h / np.maximum(val, 1e-30))):
            p_.stairs(np.where(msk & above, h_, np.nan), e, color="k", ls=":",
                      lw=LW["fo"], label=(FO_LABEL if p_ is a else None))
            p_.stairs(np.where(msk & ~above, h_, np.nan), e, color="k", ls=":",
                      lw=1.4, alpha=0.22)
    for lbl, h, col, ls, neg in series:
        lw = 3.2 if "MaxEnt" in lbl else 1.9
        leg = (rf"{lbl} ({med(h):.1f}\%)" if neg is None
               else rf"{lbl} ({med(h):.1f}\%, {neg}\% $w<0$)")
        a.stairs(np.where(msk, h, np.nan), e, color=col, ls=ls, lw=lw, label=leg)
        r.stairs(np.where(msk, h / np.maximum(val, 1e-30), np.nan), e, color=col, ls=ls, lw=lw)
    r.fill_between(ctr[msk], (1 - rel)[msk], (1 + rel)[msk], color=C["band"], alpha=0.5, step="mid")
    r.axhline(1, color="k", lw=0.8)
    # the pT seam maps onto phi* via phi* ~ pT/m  =>  mark it
    xs = XM / 91.1876
    for p_ in (a, r):
        p_.axvline(xs, color=C["seam"], lw=1.8, ls="--")
    a.set_xscale("log"); a.set_yscale("log"); r.set_xscale("log")
    # axes-fraction coords: immune to the y-scale being set after this call.
    # (Reading get_ylim() on the still-linear axis put the label at a negative y,
    # which is off-scale once the axis goes log -- bbox="tight" then grew the
    # canvas to 56000 pt to contain it.)
    a.text(xs * 1.10, 0.04, rf"$x_{{\rm match}}\!\to\!\phi^*\!\simeq\!{xs:.2f}$",
           transform=a.get_xaxis_transform(), color=C["seam"], fontsize=13,
           rotation=90, va="bottom")
    a.tick_params(labelbottom=False); r.set_ylim(0.80, 1.20)
    a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}\phi^*_\eta$")
    r.set_ylabel(r"ratio to data"); r.set_xlabel(r"$\phi^*_\eta$")
    x0 = e[e > 0].min(); a.set_xlim(x0, e[-1]); r.set_xlim(x0, e[-1])
    a.legend(loc="lower left", bbox_to_anchor=(0.015, 0.015), handlelength=1.4,
             labelspacing=0.28, borderaxespad=0.0)
    a.set_title(r"$\phi^*_\eta$ never constrained: a pure prediction of the reweighting")
    out = os.path.join(HERE, "fig_phistar_prediction.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    print("wrote", out)
    for lbl, h, *_ in series:
        print(f"  {lbl:14s} med|ratio-1| = {med(h):5.2f}%")


def cfg():
    return dict(born={"mll": {"range": (66., 116.), "map": "lin"}, "y_abs": {"range": (0., 2.4), "map": "lin"}},
                recoil={"pT_ll": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                                  "profile": {"a": XM, "b": 2 * XM, "c": XHI}}},
                followers=["phistar"])


if __name__ == "__main__":
    main()
