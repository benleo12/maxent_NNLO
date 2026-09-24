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

BASE = os.path.join(os.environ.get("NNLOJET_ROOT",
        os.path.expanduser("~/nnlojet-v1.0.2")), "dy_profile_log30_hi")
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
XM, XHI, SOFT, XMAP = 30.0, 500.0, 30.0, 2500.0
# Born-level prior: final leptons + their own QED FSR photons (post-ISR-recoil,
# pre-FSR), the convention of a QCD-only fixed-order calculation.  The older
# dy_psLO_ext_* files stored only bare leptons.
PRIOR_FILES = ["dy_psLO_born_1.npz"]
NMOM = 6
# MIRROR the Fortran exactly -- eval_chebT_ptl1 uses log [27,200] and
# eval_chebT_ptl2 log [27,150].  They are NOT the same map, and using one for
# both compares moments of different observables.
PT_MAP = {"pt_l1": (27.0, 200.0), "pt_l2": (27.0, 150.0)}


def load_prior():
    """The ATLAS-fiducial Born-level prior, identical events and order to every
    other DY figure (dy_prior_atlas_v3.npz), so the shared band cache applies."""
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v3.npz")))
    w = np.asarray(P["w"], float)
    m = np.isfinite(w) & (w > 0)
    return dict(mll=np.asarray(P["mll"], float)[m], y_abs=np.abs(np.asarray(P["y_ll"], float))[m],
                pT_ll=np.asarray(P["pT_ll"], float)[m],
                pt_l1=np.asarray(P["pT_lead"], float)[m], pt_l2=np.asarray(P["pT_sub"], float)[m],
                weight=w[m])


def solve():
    seeds = common_seeds(BASE, "DY_MOMENTS", CH6)
    M = fo_moments_smooth_from_nnlojet(
        BASE, "DY_MOMENTS", CH6, seeds,
        born_tags={"mll": "mll", "y_abs": "absyz"},
        n_born=12, n_recoil=20, x_match=XM, x_hi=XMAP, soft_lo=SOFT)
    ev = load_prior()
    n = len(ev["weight"])
    idx = np.random.default_rng(0).choice(n, min(1_000_000, n), replace=False)   # same draw as the ATLAS figures
    ev = {k: v[idx] for k, v in ev.items()}
    # "bw" mirrors the Breit-Wigner map compiled into eval_chebT_mll
    cfg = dict(born={"mll": {"range": (66., 116.), "map": "bw"},
                     "y_abs": {"range": (0., 2.4), "map": "lin"}},
               recoil={"pT_ll": {"range": (SOFT, XMAP), "map": "log", "soft_lo": SOFT,
                                 "profile": {"a": XM, "b": 2 * XM, "c": XHI}}},
               followers=["pt_l1", "pt_l2"])
    res = upgrade(ev, M, cfg)
    print(f"  prior events {len(ev['weight']):,}   effN {100*res.effN:.1f}%   "
          f"closure {res.closure:.2e}   neg-wt {100*np.mean(res.weights<=0):.1f}%")
    # the same upgrade with the NNLO Z+jet recoil (Stripper) through the hook
    res2 = None
    xml = os.path.join(HERE, "ppzj-moments_NNLO.xml")
    if os.path.exists(xml):
        os.environ["DY_RECOIL_XML"] = xml
        try:
            M2 = fo_moments_smooth_from_nnlojet(
                BASE, "DY_MOMENTS", CH6, seeds,
                born_tags={"mll": "mll", "y_abs": "absyz"},
                n_born=12, n_recoil=20, x_match=XM, x_hi=XMAP, soft_lo=SOFT)
        finally:
            os.environ.pop("DY_RECOIL_XML", None)
        res2 = upgrade(ev, M2, cfg)
        print(f"  NNLO(Zj) recoil: effN {100*res2.effN:.1f}%   closure {res2.closure:.2e}")
    return ev, res, seeds, res2


def fig_moments(ev, res, seeds, res2=None):
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

        # prior moment statistical error: weighted SEM of T_n over the sample
        u_ = np.clip((np.log(np.clip(ev[key], lo_, hi_)) - np.log(lo_))
                     / (np.log(hi_) - np.log(lo_)) * 2 - 1, -1, 1)
        T_ = [np.ones_like(u_), u_]
        for _n in range(2, NMOM + 1):
            T_.append(2 * u_ * T_[-1] - T_[-2])
        wp_ = np.asarray(ev["weight"], float); sw_ = wp_.sum()
        mp_err = np.array([np.sqrt((wp_ ** 2 * (T_[n] - mp[n - 1]) ** 2).sum()) / sw_
                           for n in nn])

        # MaxEnt moment uncertainty: bootstrap spread (stat) (+) scale envelope
        # of the variant re-solves, from moment_bands (ext-prior config)
        mq_err = None
        bwf = os.path.join(HERE, "dy_band_weights.npz")
        if os.path.exists(bwf):
            BW = dict(np.load(bwf))
            if np.allclose(BW["central"], res.weights, rtol=1e-6, atol=0):
                mb = np.array([chebyshev_moment(ev[key], w_, NMOM, lo_, hi_, "log")
                               for w_ in BW["boot_w"]]).std(0, ddof=1)
                ms_ = np.array([chebyshev_moment(ev[key], w_, NMOM, lo_, hi_, "log")
                                for w_ in BW["scale_w"]])
                half = 0.5 * (ms_.max(0) - ms_.min(0))
                mq_err = np.hypot(mb, half)
            else:
                print("    WARNING: dy_band_weights.npz stale; no MaxEnt bars")
        mq2 = mq2_err = None
        if res2 is not None:
            mq2 = chebyshev_moment(ev[key], res2.weights, NMOM, lo_, hi_, "log")
            bwf2 = os.path.join(HERE, "dy_band_weights_nnloZj.npz")
            if os.path.exists(bwf2):
                BW2 = dict(np.load(bwf2))
                if np.allclose(BW2["central"], res2.weights, rtol=1e-6, atol=0):
                    mb2 = np.array([chebyshev_moment(ev[key], w_, NMOM, lo_, hi_, "log")
                                    for w_ in BW2["boot_w"]]).std(0, ddof=1)
                    ms2 = np.array([chebyshev_moment(ev[key], w_, NMOM, lo_, hi_, "log")
                                    for w_ in BW2["scale_w"]])
                    mq2_err = np.hypot(mb2, 0.5 * (ms2.max(0) - ms2.min(0)))
                else:
                    print("    WARNING: dy_band_weights_nnloZj.npz stale; no bars on the NNLO(Zj) points")

        fig, ax = plt.subplots(2, 1, figsize=(6.8, 7.4),
                               gridspec_kw={"height_ratios": [2.0, 1.15], "hspace": 0.08})
        a, r = ax
        a.errorbar(nn, fo, yerr=er, fmt="s", color=C["fo"], ms=8, lw=1.6, capsize=4,
                   label=r"fixed order (NNLO)", zorder=5)
        a.errorbar(nn, mp, yerr=mp_err, fmt="o", color=C["prior"], ms=9,
                   lw=LW["prior"], capsize=3, label=r"PS+LO prior")
        a.errorbar(nn, mq, yerr=mq_err, fmt="o", color=C["maxent"], ms=9,
                   lw=LW["maxent"], capsize=3,
                   label=(r"MaxEnt, NLO$(Zj)$ recoil (predicted)" if mq2 is not None else r"MaxEnt (predicted)"))
        if mq2 is not None:
            a.errorbar(nn + 0.15, mq2, yerr=mq2_err, fmt="o", color=C["maxent_nnlo"], ms=9,
                       lw=LW["maxent_nnlo"], capsize=3, label=r"MaxEnt, NNLO$(Zj)$ recoil (predicted)")
        a.axhline(0, color="k", lw=0.8)
        a.set_ylabel(rf"$\langle T_n({lab})\rangle$")   # lab is bare math
        a.set_title(rf"${lab}$ predicted at NNLO")
        a.tick_params(labelbottom=False)
        a.legend(loc="best", fontsize=13)
        # pulls: (X - FO) / sigma_FO, the honest measure given FO has errors
        r.errorbar(nn, (mp - fo) / np.maximum(er, 1e-12),
                   yerr=mp_err / np.maximum(er, 1e-12), fmt="o", color=C["prior"],
                   ms=9, ls=LS["prior"], lw=LW["prior"], capsize=3)
        r.errorbar(nn, (mq - fo) / np.maximum(er, 1e-12),
                   yerr=(mq_err / np.maximum(er, 1e-12) if mq_err is not None else None),
                   fmt="o", color=C["maxent"], ms=9, ls=LS["maxent"],
                   lw=LW["maxent"], capsize=3)
        if mq2 is not None:
            r.errorbar(nn + 0.15, (mq2 - fo) / np.maximum(er, 1e-12),
                       yerr=(mq2_err / np.maximum(er, 1e-12) if mq2_err is not None else None),
                       fmt="o", color=C["maxent_nnlo"], ms=9, ls=LS["maxent_nnlo"],
                       lw=LW["maxent_nnlo"], capsize=3)
        r.axhspan(-1, 1, color=C["band"], alpha=0.45)
        r.axhline(0, color="k", lw=0.8)
        r.set_xlabel(r"Chebyshev order $n$")
        r.set_ylabel(r"$(\,\cdot\,-\,\mathrm{FO})/\sigma_{\mathrm{FO}}$")
        r.set_xticks(nn)
        out = os.path.join(HERE, f"fig_nnlo_{key}_moments.pdf")
        fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
        print("wrote", out)
        print(f"    {lab}: prior pull {np.abs((mp-fo)/np.maximum(er,1e-12)).mean():6.1f}"
              f"   MaxEnt pull {np.abs((mq-fo)/np.maximum(er,1e-12)).mean():6.1f}"
              + (f"   MaxEnt NNLO(Zj) recoil pull {np.abs((mq2-fo)/np.maximum(er,1e-12)).mean():6.1f}" if mq2 is not None else ""))


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
    ev, res, seeds, res2 = solve()
    fig_moments(ev, res, seeds, res2)
    # |y_ll| lives in fig_dy_spectra.py (fig_dy_yll); duplicating it here
    # would put the same observable in two figures from two solves.


if __name__ == "__main__":
    main()
