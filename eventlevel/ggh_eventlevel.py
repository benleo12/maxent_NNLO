#!/usr/bin/env python3
r"""gg -> H upgraded with EVENT-LEVEL NNLOJET moments  (single-author JHEP).

Constrained : |y_H| (Born, lin [0,4]) and pT_H (recoil, log [1,1000] with the
              smooth profile [30,60]->1000, exactly as booked in NNLOJET).
Reference   : the fixed-order pT_H and |y_H| distributions.

Produces the first gg->H figures for this project, in the common publication
style, with ratio sub-panels.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import use_pub_style
use_pub_style(base=17)
from maxent_upgrade import upgrade, check_seam
from nnlojet_moments import fo_moments_smooth_from_nnlojet, common_seeds, _load

GDIR = "/Users/user/nnlojet-v1.0.2/ggh_moments"
RUN, PREFIX = "GGH_MOMENTS", "H"
CH = ["LO", "R", "V"]
# The profile is COMPILED INTO NNLOJET (eval_w_pth: pa=30, pb=60, pc=pd=1000),
# and the FO moments are <T_n w> with THAT w -- so these must mirror the
# Fortran exactly.  Moving them here alone imposes moments built with one
# weight function against features built with another (it costs a factor 10
# in closure).  check_seam() flags that gg->H should really sit at 37 GeV;
# acting on it means editing EvalFuncs.f90 and re-running NNLOJET.
XM, XB, XHI, SOFT = 30.0, 60.0, 1000.0, 1.0
Q_HARD = 125.0
YHI = 4.0


def fo_curve(tag, seeds, channels=("R",)):
    """FO reference density (channels with pT>0 for the recoil), pooled."""
    lo = hi = None; tot = None
    for s in seeds:
        for ch in channels:
            r = _load(os.path.join(GDIR, f"ch_{ch}", f"{PREFIX}.{RUN}.{ch}.{tag}.s{s}.dat"))
            if r is None: continue
            lo, _, hi, v, _ = r
            tot = v[:, 0].copy() if tot is None else tot + v[:, 0]
    return (lo, hi, tot) if tot is not None else None


def dens(x, w, e):
    h, _ = np.histogram(x, e, weights=w / w.sum()); return h / np.diff(e)


def main():
    check_seam(XM, Q_HARD, label="gg->H")
    seeds = common_seeds(GDIR, RUN, CH, prefix=PREFIX)
    print(f"gg->H FO seeds usable: {seeds}")
    M = fo_moments_smooth_from_nnlojet(
        GDIR, RUN, CH, seeds, born_tags={"y_abs": "yh"},
        n_born=6, n_recoil=12, x_match=XM, x_hi=XHI, soft_lo=SOFT,
        recoil_cfg_name="pT_H", norm_born="norm_born",
        w0="prof_wpt_0", wtag="prof_wpt", prefix=PREFIX)
    print("  <T_n(|y_H|)> :", " ".join(f"{v:+.4f}" for v in M["born"]["y_abs"]["values"]))
    rc = M["recoil"]["pT_H"]
    print("  <T_n(pT_H)>_w:", " ".join(f"{v:+.4f}" for v in rc["window_values"][:6]), "...")
    print(f"  w-rate R = {rc['rate']:.4f}")

    parts = [dict(np.load(os.path.join(HERE, f), allow_pickle=True))
             for f in ("ggh_prior_s1.npz", "ggh_prior_s2.npz")
             if os.path.exists(os.path.join(HERE, f))]
    g = lambda k: np.concatenate([np.asarray(p[k], float) for p in parts])
    ev = dict(pT_H=g("pT_H"), y_abs=np.abs(g("y_H")), weight=g("weight"))
    m = np.isfinite(ev["pT_H"]) & np.isfinite(ev["weight"]) & (ev["weight"] > 0)
    ev = {k: v[m] for k, v in ev.items()}
    n = len(ev["weight"])
    idx = np.random.default_rng(0).choice(n, min(1_200_000, n), replace=False)
    ev = {k: v[idx] for k, v in ev.items()}
    print(f"  prior events: {len(ev['weight']):,}")

    cfg = dict(born={"y_abs": {"range": (0.0, YHI), "map": "lin"}},
               recoil={"pT_H": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                                "profile": {"a": XM, "b": XB, "c": XHI}}},
               moment_selection=False)
    print("solving ...", flush=True)
    res = upgrade(ev, M, cfg)
    print(f"  effN {100*res.effN:.1f}%  closure {res.closure:.2e}  "
          f"neg-wt {100*np.mean(res.weights<=0):.1f}%")

    # ---------------- figure ----------------
    panels = [("pT_H", np.geomspace(1, 500, 30), r"$p_T^{H}$ [GeV]", True, "pth_all", ("R",)),
              ("y_abs", np.linspace(0, YHI, 25), r"$|y_H|$", False, "absyh_a", tuple(CH))]
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6), squeeze=False,
                           gridspec_kw={"height_ratios": [2.1, 1.15], "hspace": 0.07,
                                        "wspace": 0.26})
    for j, (key, e, lab, logx, fotag, fch) in enumerate(panels):
        a, r = ax[0, j], ax[1, j]
        hp = dens(ev[key], ev["weight"], e)
        hq = dens(ev[key], res.weights, e)
        ctr = np.sqrt(e[:-1] * e[1:]) if logx else 0.5 * (e[:-1] + e[1:])
        a.stairs(hp, e, color="0.55", ls="--", lw=2.0, label=r"PS+LO prior")
        a.stairs(hq, e, color="#d62728", lw=3.0, label=r"MaxEnt ($0\%\ w<0$)")
        ref = None  # set to FO below
        fc = fo_curve(fotag, seeds, fch)
        if fc is not None:
            flo, fhi, fv = fc
            fctr = np.sqrt(flo * fhi) if logx else 0.5 * (flo + fhi)
            good = fv > 0
            # normalise FO onto the plotted region for a shape comparison
            # normalise the FO curve over the region where it is TRUSTED
            # (the window for the recoil; the full range for a Born variable) --
            # normalising over the full pT range would be dominated by the
            # low-pT region where fixed order diverges and is unphysical.
            if key == "pT_H":
                sel = good & (fctr >= XM) & (fctr < min(XHI, e[-1]))
                inwin = (ev[key] >= XM) & (ev[key] < min(XHI, e[-1]))
                target = float(res.weights[inwin].sum() / res.weights.sum())
            else:
                sel = good & (fctr >= e[0]) & (fctr < e[-1])
                target = float(((hq * np.diff(e))[np.isfinite(hq)]).sum())
            if sel.sum() > 2:
                scale = target / (fv[sel] * (fhi - flo)[sel]).sum()
                fe = np.concatenate([flo[sel][:1], fhi[sel]])
                a.stairs(fv[sel] * scale, fe, color="k", ls=":", lw=2.2, label=r"fixed order")
                fi = np.interp(ctr, fctr[sel], fv[sel] * scale, left=np.nan, right=np.nan)
                ref = fi          # fixed order is the ratio denominator
                r.stairs(fi / ref, e, color="k", ls=":", lw=2.0)
        if ref is None: ref = np.maximum(hq, 1e-30)
        r.stairs(hp / ref, e, color="0.55", ls="--", lw=1.8)
        r.stairs(hq / ref, e, color="#d62728", lw=2.4)
        r.axhline(1, color="k", lw=0.8)
        if logx:
            a.set_xscale("log"); r.set_xscale("log"); a.set_yscale("log")
        if key == "pT_H":
            for p in (a, r):
                p.axvspan(XM, min(XHI, e[-1]), color="#ffd24d", alpha=0.13)
                p.axvline(XM, color="0.35", lw=2.0, ls="--")
        a.tick_params(labelbottom=False)
        a.set_title(lab + (r"  \small(recoil, constrained above the seam)" if key == "pT_H"
                           else r"  \small(Born, constrained)"), fontsize=15)
        r.set_xlabel(lab); r.set_ylim(0.5, 1.5)
        if j == 0:
            a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}X$")
            r.set_ylabel(r"ratio to fixed order")
            a.legend(loc="lower left", fontsize=12)
    fig.suptitle(r"$gg\to H$ at 13 TeV, event-level NLO moments (LO+R+V)"
                 "\n" rf"\small effN $={100*res.effN:.0f}\%$, closure $={res.closure:.1e}$, "
                 r"$0\%$ negative weights", y=1.03)
    out = os.path.join(HERE, "fig_ggh_eventlevel.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
