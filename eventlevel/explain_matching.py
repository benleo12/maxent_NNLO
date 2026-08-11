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
from pubstyle import use_pub_style
use_pub_style(base=20)
from maxent_upgrade import upgrade
from nnlojet_moments import fo_moments_smooth_from_nnlojet, common_seeds, _load

BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_poc"
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
XM, XHI, SOFT = 30.0, 500.0, 0.5
C_SH, C_FO, C_ME = "#1f77b4", "k", "#d62728"


def main():
    seeds = common_seeds(BASE, "DY_MOMENTS", CH6)
    M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, seeds,
                                       born_tags={"mll": "mll", "y_abs": "absyz"},
                                       n_born=6, n_recoil=12,
                                       x_match=XM, x_hi=XHI, soft_lo=SOFT)
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v2.npz")))
    i = np.random.default_rng(0).choice(len(P["w"]), 600000, replace=False)
    ev = dict(mll=P["mll"][i].astype(float), y_abs=np.abs(P["y_ll"][i]).astype(float),
              pT_ll=P["pT_ll"][i].astype(float), weight=P["w"][i].astype(float))
    res = upgrade(ev, M, dict(
        born={"mll": {"range": (66., 116.), "map": "lin"},
              "y_abs": {"range": (0., 2.4), "map": "lin"}},
        recoil={"pT_ll": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                          "profile": {"a": XM, "b": 2 * XM, "c": XHI}}}))
    x, wpr, wpo = ev["pT_ll"], ev["weight"], res.weights

    # FO recoil curve (only pT>0 channels populate the window)
    flo = fhi = None; tot = None
    for s_ in seeds:
        for ch in ("R", "RR", "RV"):
            r0 = _load(os.path.join(BASE, f"ch_{ch}", f"Z.DY_MOMENTS.{ch}.ptz_winfine.s{s_}.dat"))
            if r0 is None: continue
            flo, _, fhi, v, _ = r0
            tot = v[:, 0].copy() if tot is None else tot + v[:, 0]

    e = np.geomspace(2, 300, 34); ctr = np.sqrt(e[:-1] * e[1:]); bw = np.diff(e)
    d = lambda w: np.histogram(x, e, weights=w / w.sum())[0] / bw
    hpr, hpo = d(wpr), d(wpo)
    g = (tot > 0) & (fhi > flo)
    inw = (x >= XM) & (x < XHI)
    sc = float(wpo[inw].sum() / wpo.sum()) / (tot[g] * (fhi - flo)[g]).sum()
    fe, fv = np.concatenate([flo[g][:1], fhi[g]]), tot[g] * sc

    fig, ax = plt.subplots(1, 3, figsize=(19.5, 6.0))
    for a in ax:
        a.set_xscale("log"); a.set_xlim(2, 300)
        a.set_xlabel(r"$p_T^{\ell\ell}$ [GeV]")
        a.axvline(XM, color="0.45", lw=1.6, ls="--")

    # -------- 1. ingredients
    a = ax[0]
    below = ctr < XM
    a.plot(ctr[below], hpr[below], color=C_SH, lw=3.4, label=r"parton shower")
    a.plot(ctr[~below], hpr[~below], color=C_SH, lw=2.0, alpha=0.30)
    a.stairs(fv, fe, color=C_FO, ls=":", lw=3.0, label=r"fixed order")
    a.set_yscale("log"); a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}p_T$")
    a.set_title(r"1.\ \ each trusted in its own region")
    a.legend(loc="lower left")

    # -------- 2. result
    a = ax[1]
    a.plot(ctr, hpr, color="0.6", lw=2.2, ls="--", label=r"prior")
    a.stairs(fv, fe, color=C_FO, ls=":", lw=2.6, label=r"fixed order")
    a.plot(ctr, hpo, color=C_ME, lw=3.6, label=r"reweighted")
    a.set_yscale("log"); a.set_title(r"2.\ \ shower below, fixed order above")
    a.legend(loc="lower left")

    # -------- 3. the ratio
    a = ax[2]
    r = hpo / np.maximum(hpr, 1e-30)
    z = float(np.mean(r[(ctr < XM) & (hpr > 0)]))
    a.plot(ctr, r, color=C_ME, lw=3.6)
    a.axhline(1.0, color="k", lw=1.0)
    a.axhline(z, color=C_SH, lw=2.2, ls="--")
    a.set_ylim(0.4, 2.4)
    a.set_ylabel(r"reweighted\,/\,prior")
    a.set_title(r"3.\ \ flat below the seam $\Rightarrow$ shape kept")
    a.text(3.0, z + 0.10, rf"$1/Z={z:.2f}$", color=C_SH, fontsize=19)

    out = os.path.join(HERE, "fig_explain_matching.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    print(f"wrote {out}   (1/Z={z:.3f}, effN={100*res.effN:.0f}%)")


if __name__ == "__main__":
    main()
