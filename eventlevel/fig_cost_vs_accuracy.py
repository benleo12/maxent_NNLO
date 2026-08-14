#!/usr/bin/env python3
r"""Accuracy against statistical cost.

The practical claim of the method in one plot: the reweighted sample is more
accurate than every matched generator on an observable it never saw, and it is
an order of magnitude cheaper in generated events, because it carries no
negative weights.

  x   generated events needed per effective event, N / N_eff, with
      N_eff = (sum w)^2 / sum w^2.  Negative weights are what push this up.
  y   median |shape/data - 1| on phi*_eta against ATLAS 1912.02844.

Both axes are "lower is better", so the useful corner is the bottom left.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import use_pub_style, C
use_pub_style(base=19)

GENS = [("MaxEnt",  "maxent",  None),
        ("MiNNLO",  "minnlo",  "dy_minnlo_atlas_v2.npz"),
        ("MC@NLO",  "mcatnlo", "dy_mcatnlo_atlas.npz"),
        ("POWHEG",  "powheg",  "dy_powheg_atlas.npz")]


def dens(x, w, e):
    h, _ = np.histogram(x, e, weights=w / w.sum())
    return h / np.diff(e)


def main():
    from maxent_upgrade import upgrade
    from nnlojet_moments import fo_moments_smooth_from_nnlojet, common_seeds
    BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_poc"
    CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
    XM, XHI, SOFT = 30.0, 500.0, 0.5

    D = dict(np.load(os.path.join(HERE, "atlas_phistar_born.npz")))
    e = np.concatenate([D["lo"][:1], D["hi"]]); val = D["val"]; bw = np.diff(e)
    msk = val > 0

    def dev(x, w):
        h = dens(np.asarray(x, float), np.asarray(w, float), e)
        m = msk & (h > 0)
        hh = h[m] / (h[m] * bw[m]).sum(); dd = val[m] / (val[m] * bw[m]).sum()
        return 100 * np.median(np.abs(hh / dd - 1))

    # --- our sample -------------------------------------------------------
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v2.npz")))
    n = len(P["w"]); idx = np.random.default_rng(0).choice(n, min(1_000_000, n), replace=False)
    ev = dict(mll=P["mll"][idx].astype(float), y_abs=np.abs(P["y_ll"][idx]).astype(float),
              pT_ll=P["pT_ll"][idx].astype(float), phistar=P["phistar"][idx].astype(float),
              weight=P["w"][idx].astype(float))
    M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6,
                                       common_seeds(BASE, "DY_MOMENTS", CH6),
                                       born_tags={"mll": "mll", "y_abs": "absyz"},
                                       n_born=6, n_recoil=12,
                                       x_match=XM, x_hi=XHI, soft_lo=SOFT)
    res = upgrade(ev, M, dict(
        born={"mll": {"range": (66., 116.), "map": "lin"},
              "y_abs": {"range": (0., 2.4), "map": "lin"}},
        recoil={"pT_ll": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                          "profile": {"a": XM, "b": 2 * XM, "c": XHI}}},
        followers=["phistar"]))

    pts = []
    for lbl, key, fn in GENS:
        if fn is None:
            w = np.asarray(res.weights, float); x = ev["phistar"]
        else:
            G = dict(np.load(os.path.join(HERE, fn)))
            w = np.asarray(G["w"], float); x = np.asarray(G["phistar"], float)
        neff = (w.sum() ** 2) / (len(w) * (w ** 2).sum())
        pts.append((lbl, key, 1.0 / neff, dev(x, w), 100 * np.mean(w < 0)))
        print(f"  {lbl:8s} N/N_eff = {1/neff:6.2f}   phi* {pts[-1][3]:5.2f}%   "
              f"neg-wt {pts[-1][4]:4.1f}%")

    fig, a = plt.subplots(figsize=(7.4, 6.4))
    for lbl, key, cost, acc, neg in pts:
        a.plot(cost, acc, "o", color=C[key], ms=17, zorder=5,
               markeredgecolor="white", markeredgewidth=1.4)
        dx = 1.10 if key != "minnlo" else 0.62
        a.annotate(lbl, (cost, acc), xytext=(cost * dx, acc * 1.045),
                   color=C[key], fontsize=18,
                   ha="left" if key != "minnlo" else "right")
    a.set_xscale("log")
    a.set_xlabel(r"generated events per effective event,\ \ $N/N_{\rm eff}$")
    a.set_ylabel(r"$\phi^*_\eta$:\ \ median $|$prediction$/$data$-1|$ [\%]")
    a.set_title(r"Accuracy and statistical cost")
    a.set_xlim(0.85, 30.0)
    a.set_ylim(0.4, 4.3)
    out = os.path.join(HERE, "fig_cost_vs_accuracy.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
