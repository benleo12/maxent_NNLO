#!/usr/bin/env python3
r"""Companion to the phi* plot, answering the obvious referee objection:
"phi* is basically pT, so of course it works."

We therefore predict observables that are DECORRELATED from the recoil we
constrained (Spearman |rho| vs pT_ll, measured on the prior):

    phi*                0.83   <- essentially the recoil
    |Delta eta(l1,l2)|  0.10
    |eta| leading lep   0.06
    |cos theta*| (CS)   0.06

None of these lepton-angle observables is constrained: the FO moments are
imposed ONLY on {m_ll, |y_ll|, pT_ll}.  They are pure predictions, compared
head-to-head with the matched generators (which all store lepton four-vectors,
so every curve is built identically).
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
from maxent_upgrade import upgrade
from nnlojet_moments import fo_moments_smooth_from_nnlojet, common_seeds, _load

BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_poc"
XM, XHI, SOFT = 30.0, 500.0, 0.5
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
SEEDS = None
PRIOR_FILES = [f"dy_psLO_ext_{i}.npz" for i in (1, 2, 3, 4)]


def ptyphi(v):
    px, py, pz, E = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    return np.hypot(px, py), 0.5 * np.log((E + pz) / np.maximum(E - pz, 1e-12)), np.arctan2(py, px)


def build(fn):
    """Load a showered sample -> dict of observables + weight, fiducial cut applied."""
    z = np.load(os.path.join(HERE, fn), allow_pickle=True)
    lp = np.asarray(z["l_plus"], float); lm = np.asarray(z["l_minus"], float)
    mll = np.asarray(z["mll"], float); pT = np.asarray(z["pT_ll"], float)
    yll = np.asarray(z["y_ll"], float)
    w = np.asarray(z["weight"], float) if "weight" in z.files else np.ones(len(mll))
    ptp, yp, _ = ptyphi(lp); ptm, ym, _ = ptyphi(lm)
    m = (ptp > 27) & (ptm > 27) & (np.abs(yp) < 2.5) & (np.abs(ym) < 2.5) \
        & (mll > 66) & (mll < 116) & np.isfinite(pT) & np.isfinite(w)
    p1p = (lp[:, 3] + lp[:, 2]) / np.sqrt(2); p1m = (lp[:, 3] - lp[:, 2]) / np.sqrt(2)
    p2p = (lm[:, 3] + lm[:, 2]) / np.sqrt(2); p2m = (lm[:, 3] - lm[:, 2]) / np.sqrt(2)
    pz = lp[:, 2] + lm[:, 2]
    cs = np.abs(np.sign(pz) * 2 * (p1p * p2m - p1m * p2p)
                / np.maximum(mll * np.sqrt(mll ** 2 + pT ** 2), 1e-12))
    return dict(mll=mll[m], y_abs=np.abs(yll)[m], pT_ll=pT[m],
                deta=np.abs(yp - ym)[m],
                eta_lead=np.where(ptp >= ptm, np.abs(yp), np.abs(ym))[m],
                cts=np.clip(cs, 0, 1)[m], weight=w[m])


def cat(ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


def dens(x, w, e):
    h, _ = np.histogram(x, e, weights=w / w.sum()); return h / np.diff(e)


OBS = [("deta", r"$|\Delta\eta_{\ell\ell}|$", np.linspace(0, 4.0, 26), 0.10),
       ("eta_lead", r"$|\eta_{\ell,\mathrm{lead}}|$", np.linspace(0, 2.5, 26), 0.06),
       ("cts", r"$|\cos\theta^*_{\mathrm{CS}}|$", np.linspace(0, 1.0, 26), 0.06)]


def main():
    print("loading prior ...", flush=True)
    ev = cat([build(f) for f in PRIOR_FILES if os.path.exists(os.path.join(HERE, f))])
    n = len(ev["weight"])
    idx = np.random.default_rng(0).choice(n, min(1_200_000, n), replace=False)
    ev = {k: v[idx] for k, v in ev.items()}
    print(f"  prior events (fiducial): {len(ev['weight']):,}", flush=True)

    M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, (SEEDS or common_seeds(BASE, 'DY_MOMENTS', CH6)),
                                       born_tags={"mll": "mll", "y_abs": "absyz"},
                                       n_born=6, n_recoil=12, x_match=XM, x_hi=XHI, soft_lo=SOFT)
    cfg = dict(born={"mll": {"range": (66., 116.), "map": "lin"},
                     "y_abs": {"range": (0., 2.4), "map": "lin"}},
               recoil={"pT_ll": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                                 "profile": {"a": XM, "b": 2 * XM, "c": XHI}}},
               followers=[o for o, _, _, _ in OBS])
    print("solving ...", flush=True)
    res = upgrade(ev, M, cfg)
    print(f"  effN={100*res.effN:.0f}%  closure={res.closure:.1e}  neg-wt={100*np.mean(res.weights<=0):.0f}%", flush=True)

    gens = {}
    for lbl, fn, col, neg in [("MiNNLO", "dy_minnlo_s1_showered.npz", C["minnlo"], 23),
                              ("POWHEG", "dy_psNLO_powheg_ct18_fixed.npz", C["powheg"], 1)]:
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            try:
                gens[lbl] = (build(fn), col, neg)
                print(f"  {lbl}: {len(gens[lbl][0]['weight']):,} events")
            except Exception as e:
                print(f"  {lbl}: skipped ({e})")

    fig, ax = plt.subplots(2, len(OBS), figsize=(5.6 * len(OBS), 7.2), squeeze=False,
                           gridspec_kw={"height_ratios": [2.1, 1.15], "hspace": 0.07,
                                        "wspace": 0.28})
    for j, (key, lab, e, rho) in enumerate(OBS):
        a, r = ax[0, j], ax[1, j]
        hp = dens(ev[key], ev["weight"], e)
        hq = dens(ev[key], res.weights, e)
        # No fixed-order curve is drawn on these three panels, and that is a
        # statement about the fixed-order calculation rather than an omission.
        # |eta_lead| IS booked (abs_yl1) and its INTEGRAL is exact -- it
        # reproduces norm_born channel by channel, seed scatter 5e-4.  But the
        # differential distribution is not resolved: at NNLO the real-emission
        # and subtraction terms land in DIFFERENT |eta_lead| bins, so per-bin
        # values are ~300x the physical density with relative error ~1 and
        # cancel only over the full range (40 seeds: rel = 1.06 at 25 bins,
        # 1.08 merged to 5, <5e-4 at 1).  phi*_eta is immune because phi* > 0
        # requires a real emission, so LO and V do not contribute at all and
        # there is no large cancellation (rel = 0.011).
        #
        # This is exactly why the moment interface is built on integrals:
        # moments converge where the differential spectrum does not.
        fo_h = None
        if False:  # kept for reference; re-enable with far higher FO statistics
            lo = hi = None; tot = None; var = None
            sds = common_seeds(BASE, "DY_MOMENTS", CH6, tag="yl1_a")
            for s_ in sds:
                for ch in CH6:
                    r0 = _load(os.path.join(BASE, f"ch_{ch}", f"Z.DY_MOMENTS.{ch}.yl1_a.s{s_}.dat"))
                    if r0 is None: continue
                    lo, _, hi, v, er = r0
                    tot = v[:, 0].copy() if tot is None else tot + v[:, 0]
                    var = er[:, 0]**2 if var is None else var + er[:, 0]**2
            if tot is not None:
                # The R-V cancellation leaves large errors in NNLOJET's native
                # fine bins, so merge adjacent bins before judging significance:
                # the moments are integrals and do not care about the binning,
                # and this is only a reference curve.  Merging K bins buys a
                # factor sqrt(K) on the relative error.
                K = 5
                nb = (len(tot) // K) * K
                glo = lo[:nb].reshape(-1, K)[:, 0]
                ghi = hi[:nb].reshape(-1, K)[:, -1]
                gtot = (tot[:nb] * np.diff(np.stack([lo[:nb], hi[:nb]]), axis=0)[0]
                        ).reshape(-1, K).sum(1)
                gvar = (var[:nb] * np.diff(np.stack([lo[:nb], hi[:nb]]), axis=0)[0] ** 2
                        ).reshape(-1, K).sum(1)
                gw = ghi - glo
                gd = (gtot > 0) & (gw > 0) & (np.sqrt(gvar) / np.maximum(gtot, 1e-300) < 0.15)
                n_drop = int(((gtot > 0) & ~gd).sum())
                print(f"    |eta_lead| FO: {gd.sum()}/{len(gtot)} merged bins kept "
                      f"({len(sds)} seeds, {n_drop} dropped)")
                if gd.sum() > 2:
                    # SAME EDGES as every other line on the panel
                    dens_g = gtot[gd] / gw[gd]
                    fo_h = rebin_density(glo[gd], ghi[gd], dens_g, e)
                    fo_h = fo_h / np.nansum(fo_h * np.diff(e))
                    a.stairs(fo_h, e, color=C["fo"], ls=":", lw=LW["fo"],
                             label=r"fixed order (NNLO)")
        ref = np.maximum(hp, 1e-30)   # no data: ratio to the prior
        a.stairs(hp, e, color=C["prior"], ls="--", lw=2.0, label=r"PS+LO prior")
        a.stairs(hq, e, color=C["maxent"], lw=3.0, label=r"MaxEnt ($0\%\ w<0$)")
        r.stairs(hp / ref, e, color=C["prior"], ls="--", lw=1.8)
        r.stairs(hq / ref, e, color=C["maxent"], lw=2.4)
        if fo_h is not None:
            r.stairs(fo_h / ref, e, color="k", ls=":", lw=2.0)
        for lbl, (g, col, neg) in gens.items():
            hg = dens(g[key], g["weight"], e)
            a.stairs(hg, e, color=col, lw=1.9, label=rf"{lbl} (${neg}\%\ w<0$)")
            r.stairs(hg / ref, e, color=col, lw=1.8)
        r.axhline(1, color="k", lw=0.8)
        r.set_ylim(0.80, 1.30); r.set_xlabel(lab)
        a.tick_params(labelbottom=False)
        a.set_title(rf"{lab}:  $|\rho_{{\rm S}}(p_T^{{\ell\ell}})| = {rho:.2f}$")
        if j == 0:
            a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}X$")
            r.set_ylabel(r"ratio to PS+LO prior")
            a.legend(loc="lower center", labelspacing=0.3, fontsize=12)
    fig.suptitle(r"Predictions for observables \emph{decorrelated} from the constrained recoil "
                 r"($\phi^*_\eta$ has $|\rho_{\rm S}|=0.83$ by comparison)", y=1.045)
    # Say why there is no fixed-order curve here, rather than leave it missing.
    fig.text(0.5, 0.975, r"no fixed-order curve: at NNLO the real and subtraction terms "
                         r"populate different bins of these Born angular variables, so the "
                         r"\emph{differential} spectrum is unresolved "
                         r"(rel.\ err.\ $\simeq1$) while its integral is exact ($<5\times10^{-4}$)."
                         r"  MC@NLO is absent: the stored sample keeps no lepton four-vectors",
             ha="center", va="top", fontsize=12, color=C["seam"])
    out = os.path.join(HERE, "fig_decorrelated_prediction.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
