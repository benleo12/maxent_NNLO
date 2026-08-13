#!/usr/bin/env python3
r"""The main Drell-Yan figure, rebuilt from EVENT-LEVEL moments in the common style.

Replaces the stale FIG_atlas_nnlo / fig_dy_nnlo / fig_compare5: ATLAS 1912.02844
data, the LO+PS prior, the MaxEnt upgrade (event-level NNLO moments, 0% negative
weights) and every matched generator we have (MiNNLO, MC@NLO, POWHEG), for the
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
from nnlojet_moments import fo_moments_smooth_from_nnlojet, common_seeds, _load, oriented_fo_curve

BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_poc"
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
# mirrors eval_w_ptz (pa=30, pb=60, pc=pd=500)
XM, XHI, SOFT = 30.0, 500.0, 0.5
Q_HARD = 91.1876

GENS = [("MiNNLO", "dy_minnlo_atlas_v2.npz", C["minnlo"], 23),
        ("MC@NLO", "dy_mcatnlo_atlas.npz", C["mcatnlo"], 5),
        ("POWHEG", "dy_powheg_atlas.npz", C["powheg"], 1)]
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
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v2.npz")))
    n = len(P["w"]); idx = np.random.default_rng(0).choice(n, min(1_500_000, n), replace=False)
    ev = dict(mll=P["mll"][idx].astype(float), y_abs=np.abs(P["y_ll"][idx]).astype(float),
              pT_ll=P["pT_ll"][idx].astype(float), phistar=P["phistar"][idx].astype(float),
              weight=P["w"][idx].astype(float))
    cfg = dict(born={"mll": {"range": (66., 116.), "map": "lin"},
                     "y_abs": {"range": (0., 2.4), "map": "lin"}},
               recoil={"pT_ll": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                                 "profile": {"a": XM, "b": 2 * XM, "c": XHI}}},
               followers=["phistar"])
    print("solving ...", flush=True)
    res = upgrade(ev, M, cfg)
    print(f"  effN {100*res.effN:.1f}%  closure {res.closure:.2e}  "
          f"neg-wt {100*np.mean(res.weights<=0):.1f}%")

    fig, ax = plt.subplots(2, len(PANELS), figsize=(6.9 * len(PANELS), 8.4), squeeze=False,
                           gridspec_kw={"height_ratios": [2.2, 1.15], "hspace": 0.07,
                                        "wspace": 0.24})
    summary = {}
    for j, (key, dfile, lab, role) in enumerate(PANELS):
        a, r = ax[0, j], ax[1, j]
        D = dict(np.load(os.path.join(HERE, dfile)))
        lo, hi = np.asarray(D["lo"], float), np.asarray(D["hi"], float)
        val, err = np.asarray(D["val"], float), np.asarray(D["err"], float)
        e = np.concatenate([lo[:1], hi]); ctr = np.asarray(D["center"], float)
        m = val > 0
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
        r.stairs(np.where(m, hq / np.maximum(val, 1e-30), np.nan), e, color=C["maxent"], lw=2.6)
        summary[key] = {"prior": med(hp), "MaxEnt": med(hq)}
        for lbl, fn, col, neg in GENS:
            p = os.path.join(HERE, fn)
            if not os.path.exists(p):
                continue
            G = dict(np.load(p))
            if key not in G:
                continue
            hg = dens(np.asarray(G[key], float), np.asarray(G["w"], float), e)
            a.stairs(np.where(m, hg, np.nan), e, color=col, lw=1.9,
                     label=rf"{lbl} ({med(hg):.1f}\%, ${neg}\%\,w<0$)")
            r.stairs(np.where(m, hg / np.maximum(val, 1e-30), np.nan), e, color=col, lw=1.7)
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
                r.stairs(np.where(m & ab, rr, np.nan), e, color=C["fo"], ls=":", lw=2.2)
                r.stairs(np.where(m & ~ab, rr, np.nan), e, color=C["fo"], ls=":",
                         lw=1.3, alpha=0.22)
        if key == "pT_ll":
            flo = fhi = None; tot = None
            for s_ in seeds:
                for ch in ("R", "RR", "RV"):          # only these populate pT>0
                    r0 = _load(os.path.join(BASE, f"ch_{ch}",
                               f"Z.DY_MOMENTS.{ch}.ptz_winfine.s{s_}.dat"))
                    if r0 is None: continue
                    flo, _, fhi, v, _ = r0
                    tot = v[:, 0].copy() if tot is None else tot + v[:, 0]
            if tot is not None:
                g = (tot > 0) & (fhi > flo)
                inw = (ev[key] >= XM) & (ev[key] < XHI)
                tgt = float(res.weights[inw].sum() / res.weights.sum())
                # SAME EDGES as every other line, then normalise to the window rate
                fo = rebin_density(flo[g], fhi[g], tot[g], e)
                gd = np.isfinite(fo) & (fo > 0)
                fn = np.where(gd, fo, np.nan)
                fn = fn * (tgt / np.nansum(fn * np.diff(e)))
                a.stairs(fn, e, color=C["fo"], ls=":", lw=LW["fo"],
                         label=FO_LABEL)
                r.stairs(np.where(m, fn / np.maximum(val, 1e-30), np.nan), e,
                         color=C["fo"], ls=":", lw=2.0)
        rel = err / np.maximum(val, 1e-30)
        r.fill_between(ctr[m], (1 - rel)[m], (1 + rel)[m], color=C["band"], alpha=0.55, step="mid")
        r.axhline(1, color="k", lw=0.8)
        if key == "pT_ll":
            for p_ in (a, r):
                p_.axvspan(XM, XHI, color="#ffd24d", alpha=0.13)
                p_.axvline(XM, color=C["seam"], lw=2.0, ls="--")
        else:      # phi* : the seam maps over as phi* ~ pT/m
            for p_ in (a, r):
                p_.axvline(XM / 91.1876, color=C["seam"], lw=2.0, ls="--")
        a.set_xscale("log"); a.set_yscale("log"); r.set_xscale("log")
        x0 = e[e > 0].min(); a.set_xlim(x0, e[-1]); r.set_xlim(x0, e[-1])
        a.tick_params(labelbottom=False); r.set_ylim(0.72, 1.28)
        a.set_title(lab + rf"  \small({role})", fontsize=16)
        r.set_xlabel(lab)
        if j == 0:
            a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}X$")
            r.set_ylabel(r"ratio to data")
        a.legend(loc="lower left", fontsize=11.5, labelspacing=0.28)
    fig.suptitle(r"Drell--Yan at 13 TeV, event-level moments", y=1.02)
    out = os.path.join(HERE, "fig_dy_eventlevel.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    print("wrote", out)
    for k, v in summary.items():
        print(f"  {k}: " + "  ".join(f"{a_}={b_:.2f}%" for a_, b_ in v.items()))


if __name__ == "__main__":
    main()
