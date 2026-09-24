#!/usr/bin/env python3
r"""How the matching reweighting works, in one figure (DY, pT of the Z).

Left   : the two ingredients, each drawn only where it is trusted.
Middle : the reweighted result -- shower shape below the seam, FO above.
Right  : reweighted/prior -- flat below the seam (shape untouched, height
         rescaled so the window carries the FO rate), shaped above.
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
use_pub_style(base=20)
from maxent_upgrade import upgrade
from nnlojet_moments import fo_moments_smooth_from_nnlojet, common_seeds, _load

BASE = os.path.join(os.environ.get("NNLOJET_ROOT",
        os.path.expanduser("~/nnlojet-v1.0.2")), "dy_profile_log30_hi")
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
XM, XHI, SOFT, XMAP = 30.0, 500.0, 30.0, 2500.0
# The "parton shower" curve in panel 1 IS the prior, so it takes the prior
# colour: blue means MiNNLO on every other figure and must not mean two
# different things across the paper.
C_SH, C_FO, C_ME = C["prior"], C["fo"], C["maxent"]


def main():
    seeds = common_seeds(BASE, "DY_MOMENTS", CH6)
    M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, seeds,
                                       born_tags={"mll": "mll", "y_abs": "absyz"},
                                       n_born=12, n_recoil=20,
                                       x_match=XM, x_hi=XMAP, soft_lo=SOFT)
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v3.npz")))
    i = np.random.default_rng(0).choice(len(P["w"]), min(600000, len(P["w"])), replace=False)
    ev = dict(mll=P["mll"][i].astype(float), y_abs=np.abs(P["y_ll"][i]).astype(float),
              pT_ll=P["pT_ll"][i].astype(float), weight=P["w"][i].astype(float))
    res = upgrade(ev, M, dict(
        born={"mll": {"range": (66., 116.), "map": "bw"},
              "y_abs": {"range": (0., 2.4), "map": "lin"}},
        recoil={"pT_ll": {"range": (SOFT, XMAP), "map": "log", "soft_lo": SOFT,
                          "profile": {"a": XM, "b": 2 * XM, "c": XHI}}}))
    x, wpr, wpo = ev["pT_ll"], ev["weight"], res.weights

    # Fixed order (NLO Z+jet) normalised ABSOLUTELY to the NNLO fiducial
    # cross section and drawn everywhere, below the seam included where the
    # separate ptz_full production exists; an edge is placed at the seam so
    # the two histograms splice without a gap.
    from fo_ptll_full import fo_ptll
    e = np.unique(np.concatenate([np.geomspace(2, 300, 34), [XM]]))
    ctr = np.sqrt(e[:-1] * e[1:]); bw = np.diff(e)
    d = lambda w: np.histogram(x, e, weights=w / w.sum())[0] / bw
    hpr, hpo = d(wpr), d(wpo)
    fv, _, _, have_full, _ = fo_ptll(e, seam=XM)
    fv = np.where(np.isfinite(fv) & (fv > 0), fv, np.nan)
    print(f"  fixed order below the seam: {'drawn' if have_full else 'ABSENT (ptz_full not produced yet)'}")

    fig, ax = plt.subplots(1, 3, figsize=(19.5, 6.0))
    for a in ax:
        a.set_xscale("log"); a.set_xlim(2, 300)
        a.set_xlabel(r"$p_T^{\ell\ell}$ [GeV]")
        a.axvline(XM, color=C["seam"], lw=1.6, ls="--")

    # -------- 1. ingredients
    a = ax[0]
    # both inputs in full: the shower everywhere, fixed order everywhere it
    # is positive -- each fails visibly outside its region, no fade needed
    a.plot(ctr, hpr, color=C_SH, lw=3.4, label=r"parton shower")
    a.stairs(fv, e, color=C_FO, ls=":", lw=3.0, label=r"fixed order")
    a.set_yscale("log"); a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}p_T$")
    a.set_title(r"(a) the two inputs")
    a.legend(loc="lower left")

    # -------- 2. result
    a = ax[1]
    a.plot(ctr, hpr, color="0.6", lw=2.2, ls="--", label=r"prior")
    a.stairs(fv, e, color=C_FO, ls=":", lw=2.6, label=r"fixed order")
    a.plot(ctr, hpo, color=C_ME, lw=3.6, label=r"reweighted")
    a.set_yscale("log"); a.set_title(r"(b) reweighted sample")
    a.legend(loc="lower left")

    # -------- 3. the ratio
    a = ax[2]
    r = hpo / np.maximum(hpr, 1e-30)
    z = float(np.mean(r[(ctr < XM) & (hpr > 0)]))
    a.plot(ctr, r, color=C_ME, lw=3.6)
    a.axhline(1.0, color="k", lw=1.0)
    a.axhline(z, color="0.25", lw=2.2, ls="--")
    a.set_ylim(0.4, 2.4)
    a.set_ylabel(r"reweighted\,/\,prior")
    a.set_title(r"(c) ratio to the prior")
    a.text(3.0, z + 0.10, rf"$1/Z={z:.2f}$", color="0.25", fontsize=19)

    out = os.path.join(HERE, "fig_explain_matching.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    print(f"wrote {out}   (1/Z={z:.3f}, effN={100*res.effN:.0f}%)")


if __name__ == "__main__":
    main()
