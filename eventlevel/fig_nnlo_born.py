#!/usr/bin/env python3
r"""The NNLO figure: constraints and prediction at the SAME order.

Everything else in the Drell-Yan set is measured at NLO(Z+jet) accuracy, because
phi* and pT_ll both vanish at Born level and so need a real emission -- the
inclusive Z calculation at NNLO does not give them at NNLO.  The four observables
that ARE genuinely NNLO from that calculation, and are resolved well enough to
use, are

    m_ll     SNR n=1..6:   47 5465  118 3271  153 3163
    |y_ll|                143 1016   47   11   62   26
    pT_l1                 143   36   31   10   29   30
    pT_l2                 149   22  198   37    1   39

(The single-lepton rapidities |y_l1|, |y_l2| are NOT usable: every moment comes
out at SNR <= 2, the real and subtraction terms populating different rapidity
bins the way they do for the abs_yl1 spectrum.)

So this figure constrains {m_ll, |y_ll|} at NNLO and PREDICTS pT_l1 and pT_l2 at
NNLO -- constraints and prediction at the same order, which is the one test the
rest of the set cannot do.

Two variants, because there is no booked fixed-order histogram for the lepton
pT (only the profile moments), so a conventional spectrum has no legitimate
denominator without a rerun:

  --moments   <T_n(pT_l1)> and <T_n(pT_l2)>: prior / MaxEnt / fixed order, with
              the fixed-order errors as a band.  Uses what is actually in hand
              and shows the uncertainty a referee will ask about.
  --yll       the |y_ll| spectrum against the booked absyz_fine histogram,
              a conventional plot with ratio to fixed order.
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
use_pub_style(base=17)
from maxent_upgrade import upgrade, chebyshev_moment
from nnlojet_moments import (fo_moments_smooth_from_nnlojet, common_seeds,
                             _moment_over_seeds, _reduce, fo_curve)

BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_poc"
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
XM, XHI, SOFT = 30.0, 500.0, 0.5
PRIOR_FILES = [f"dy_psLO_ext_{i}.npz" for i in (1, 2, 3, 4)]
NMOM = 6
# MIRROR the Fortran exactly -- eval_chebT_ptl1 uses log [27,200] and
# eval_chebT_ptl2 log [27,150].  They are NOT the same map, and using one for
# both compares moments of different observables.
PT_MAP = {"pt_l1": (27.0, 200.0), "pt_l2": (27.0, 150.0)}


def load_prior():
    parts = []
    for f in PRIOR_FILES:
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        lp = np.asarray(z["l_plus"], float); lm = np.asarray(z["l_minus"], float)
        mll = np.asarray(z["mll"], float); pT = np.asarray(z["pT_ll"], float)
        yll = np.asarray(z["y_ll"], float)
        w = np.asarray(z["weight"], float)
        ptp = np.hypot(lp[:, 0], lp[:, 1]); ptm = np.hypot(lm[:, 0], lm[:, 1])
        yp = 0.5 * np.log((lp[:, 3] + lp[:, 2]) / np.maximum(lp[:, 3] - lp[:, 2], 1e-12))
        ym = 0.5 * np.log((lm[:, 3] + lm[:, 2]) / np.maximum(lm[:, 3] - lm[:, 2], 1e-12))
        m = ((ptp > 27) & (ptm > 27) & (np.abs(yp) < 2.5) & (np.abs(ym) < 2.5)
             & (mll > 66) & (mll < 116) & np.isfinite(pT) & np.isfinite(w) & (w > 0))
        parts.append(dict(mll=mll[m], y_abs=np.abs(yll)[m], pT_ll=pT[m],
                          pt_l1=np.maximum(ptp, ptm)[m], pt_l2=np.minimum(ptp, ptm)[m],
                          weight=w[m]))
    if not parts:
        sys.exit("no extended DY prior found (need dy_psLO_ext_*.npz)")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def solve():
    seeds = common_seeds(BASE, "DY_MOMENTS", CH6)
    M = fo_moments_smooth_from_nnlojet(
        BASE, "DY_MOMENTS", CH6, seeds,
        born_tags={"mll": "mll", "y_abs": "absyz"},
        n_born=NMOM, n_recoil=12, x_match=XM, x_hi=XHI, soft_lo=SOFT)
    ev = load_prior()
    n = len(ev["weight"])
    idx = np.random.default_rng(0).choice(n, min(1_500_000, n), replace=False)
    ev = {k: v[idx] for k, v in ev.items()}
    cfg = dict(born={"mll": {"range": (66., 116.), "map": "lin"},
                     "y_abs": {"range": (0., 2.4), "map": "lin"}},
               recoil={"pT_ll": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                                 "profile": {"a": XM, "b": 2 * XM, "c": XHI}}},
               followers=["pt_l1", "pt_l2"])
    res = upgrade(ev, M, cfg)
    print(f"  prior events {len(ev['weight']):,}   effN {100*res.effN:.1f}%   "
          f"closure {res.closure:.2e}   neg-wt {100*np.mean(res.weights<=0):.1f}%")
    return ev, res, seeds


def fig_moments(ev, res, seeds):
    """<T_n> of the PREDICTED lepton pT: prior / MaxEnt / fixed order + errors."""
    # bare (no $) so it can be nested inside other math, plus a display form
    obs = [("pt_l1", "prof_ptl1", r"p_T^{\ell_1}"),
           ("pt_l2", "prof_ptl2", r"p_T^{\ell_2}")]
    for key, tag, lab in obs:
        fo, er = [], []
        for n in range(1, NMOM + 1):
            m = _moment_over_seeds(BASE, "DY_MOMENTS", CH6, seeds, f"{tag}_{n}",
                                   "norm_born", "Z")
            c, st, sc, tot = _reduce(m, 0); fo.append(c); er.append(tot)
        fo, er = np.array(fo), np.array(er)
        lo_, hi_ = PT_MAP[key]
        mp = chebyshev_moment(ev[key], ev["weight"], NMOM, lo_, hi_, "log")
        mq = chebyshev_moment(ev[key], res.weights, NMOM, lo_, hi_, "log")
        nn = np.arange(1, NMOM + 1)

        fig, ax = plt.subplots(2, 1, figsize=(6.8, 7.4),
                               gridspec_kw={"height_ratios": [2.0, 1.15], "hspace": 0.08})
        a, r = ax
        a.errorbar(nn, fo, yerr=er, fmt="s", color=C["fo"], ms=8, lw=1.6, capsize=4,
                   label=r"fixed order (NNLO)", zorder=5)
        a.plot(nn, mp, "o", color=C["prior"], ms=9, ls=LS["prior"], lw=LW["prior"],
               label=r"PS+LO prior")
        a.plot(nn, mq, "o", color=C["maxent"], ms=9, lw=LW["maxent"],
               label=r"MaxEnt (predicted)")
        a.axhline(0, color="k", lw=0.8)
        a.set_ylabel(rf"$\langle T_n({lab})\rangle$")   # lab is bare math
        a.set_title(rf"${lab}$ predicted at NNLO")
        a.tick_params(labelbottom=False)
        a.legend(loc="best", fontsize=13)
        # pulls: (X - FO) / sigma_FO, the honest measure given FO has errors
        for v, k, mk in ((mp, "prior", "o"), (mq, "maxent", "o")):
            r.plot(nn, (v - fo) / np.maximum(er, 1e-12), mk, color=C[k],
                   ms=9, ls=LS[k], lw=LW[k])
        r.axhspan(-1, 1, color=C["band"], alpha=0.45)
        r.axhline(0, color="k", lw=0.8)
        r.set_xlabel(r"Chebyshev order $n$")
        r.set_ylabel(r"$(\,\cdot\,-\,\mathrm{FO})/\sigma_{\mathrm{FO}}$")
        r.set_xticks(nn)
        out = os.path.join(HERE, f"fig_nnlo_{key}_moments.pdf")
        fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
        print("wrote", out)
        print(f"    {lab}: prior pull {np.abs((mp-fo)/np.maximum(er,1e-12)).mean():6.1f}"
              f"   MaxEnt pull {np.abs((mq-fo)/np.maximum(er,1e-12)).mean():6.1f}")


def fig_yll(ev, res, seeds):
    """|y_ll| spectrum against the booked absyz_fine fixed-order histogram."""
    fc = fo_curve(BASE, "DY_MOMENTS", CH6, seeds, "absyz_fine")
    if fc is None:
        print("  absyz_fine not found"); return
    flo, fhi, fd, _ = fc
    e = np.linspace(0.0, 2.4, 25); bw = np.diff(e); ctr = 0.5 * (e[:-1] + e[1:])
    g = (fd > 0) & (fhi > flo)
    fo = rebin_density(flo[g], fhi[g], fd[g], e)
    ok = np.isfinite(fo) & (fo > 0)
    fo = np.where(ok, fo, np.nan); fo = fo / np.nansum(fo * bw)
    d = lambda w: (np.histogram(ev["y_abs"], e, weights=w / w.sum())[0] / bw)
    hp, hq = d(ev["weight"]), d(res.weights)
    hp, hq = hp / (hp * bw).sum(), hq / (hq * bw).sum()

    fig, ax = plt.subplots(2, 1, figsize=(6.8, 7.6),
                           gridspec_kw={"height_ratios": [2.1, 1.15], "hspace": 0.07})
    a, r = ax
    a.stairs(fo, e, color=C["fo"], ls=LS["fo"], lw=LW["fo"], label=r"fixed order (NNLO)")
    a.stairs(hp, e, color=C["prior"], ls=LS["prior"], lw=LW["prior"], label=r"PS+LO prior")
    a.stairs(hq, e, color=C["maxent"], lw=LW["maxent"], label=r"MaxEnt")
    a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}|y_{\ell\ell}|$")
    a.tick_params(labelbottom=False); a.legend(loc="lower left", fontsize=13)
    a.set_title(r"$|y_{\ell\ell}|$ constrained at NNLO")
    for h, k, lw_ in ((hp, "prior", 1.8), (hq, "maxent", 2.6)):
        r.stairs(h / fo, e, color=C[k], ls=LS[k], lw=lw_)
    r.axhline(1, color="k", lw=0.8); r.set_ylim(0.8, 1.2)
    r.set_xlabel(r"$|y_{\ell\ell}|$"); r.set_ylabel(r"ratio to fixed order")
    out = os.path.join(HERE, "fig_nnlo_yll.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
    print("wrote", out)
    m = np.isfinite(fo)
    print(f"    |y_ll| median |ratio-1|:  prior {100*np.median(np.abs(hp[m]/fo[m]-1)):.1f}%"
          f"   MaxEnt {100*np.median(np.abs(hq[m]/fo[m]-1)):.1f}%")


def main():
    print("solving (NNLO Born constraints) ...", flush=True)
    ev, res, seeds = solve()
    fig_moments(ev, res, seeds)
    fig_yll(ev, res, seeds)


if __name__ == "__main__":
    main()
