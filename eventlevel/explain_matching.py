#!/usr/bin/env python3
r"""A picture-book explanation of the matching reweighting, on DY pT of the Z.

3 stacked panels:
  1. the two ingredients + where each is trusted (shower below seam, FO above)
  2. the reweighted result: shape=shower below the seam, =FO above
  3. reweighted/prior: FLAT below (shape kept, height rescaled) vs shaped above
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
from maxent_upgrade import upgrade
from nnlojet_moments import fo_moments_from_nnlojet, _load

BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_poc"
CH = ["LO", "R", "V"]; SEEDS = [1, 2, 3, 4]
XM, XHI, SOFT = 30.0, 500.0, 0.5

# ---- data: prior events, a masked solve (clean below-seam shape), FO curve
P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v2.npz")))
idx = np.random.default_rng(0).choice(len(P["w"]), 400000, replace=False)
ev = dict(mll=P["mll"][idx].astype(float), y_abs=np.abs(P["y_ll"][idx]).astype(float),
          pT_ll=P["pT_ll"][idx].astype(float), weight=P["w"][idx].astype(float))
M = fo_moments_from_nnlojet(BASE, "DY_MOMENTS", CH, SEEDS, born_tags={"mll": "mll", "y_abs": "absyz"},
                            recoil_tag="ptz", recoil_cfg_name="pT_ll",
                            n_born=6, n_recoil=12, x_match=XM, x_hi=XHI, soft_lo=SOFT)
cfg = dict(born={"mll": {"range": (66., 116.), "map": "lin"}, "y_abs": {"range": (0., 2.4), "map": "lin"}},
           recoil={"pT_ll": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                             "profile": {"a": XM, "b": XM, "c": XHI}}})
res = upgrade(ev, M, cfg)
x = ev["pT_ll"]; wpr = ev["weight"]; wpo = res.weights

# FO recoil curve (channel-summed fine ptz histogram, [30,500])
lo = hi = None; acc = []
for s in SEEDS:
    c = None
    for ch in CH:
        r = _load(os.path.join(BASE, f"ch_{ch}", f"Z.DY_MOMENTS.{ch}.ptz_winfine.s{s}.dat"))
        if r is None: continue
        lo, ct, hi, v, _ = r
        c = v[:, 0].copy() if c is None else (c + v[:, 0] if v.shape[0] == c.shape[0] else c)
    if c is not None: acc.append(c)
fo_ctr = np.sqrt(lo * hi); fo_val = np.mean(acc, 0)          # dsig/dpt in [30,500]

e = np.geomspace(1, 300, 40); ctr = np.sqrt(e[:-1] * e[1:]); bw = np.diff(e)
def dens(w): h, _ = np.histogram(x, e, weights=w / w.sum()); return h / bw
hpr, hpo = dens(wpr), dens(wpo)
# scale FO to the reweighted integral over [XM,XHI] so it overlays
inwin = (x >= XM) & (x < XHI)
fo_scaled = fo_val * (wpo[inwin].sum() / wpo.sum()) / ((fo_val * (hi - lo)).sum())

fig, ax = plt.subplots(3, 1, figsize=(9.5, 13.5), gridspec_kw={"hspace": 0.30})

# ---------- Panel 1: ingredients + trust regions
a = ax[0]
below = ctr < XM
a.plot(ctr[below], hpr[below], color="#1f77b4", lw=2.6, label=r"PS+LO shower")
a.plot(ctr[~below], hpr[~below], color="#1f77b4", lw=2.6, ls=(0, (2, 2)), alpha=0.5)
a.plot(fo_ctr, fo_scaled, color="k", lw=2.4, ls=":", label=r"fixed order (NNLOJET)")
a.axvline(XM, color="0.3", lw=1.2)
a.axvspan(XM, XHI, color="#ffd24d", alpha=0.16)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1, 300); a.grid(alpha=0.25, which="both")
a.set_title(r"1.\ Two ingredients: each trusted in its own region", fontsize=13, loc="left")
a.text(3.5, hpr[np.argmin(np.abs(ctr-3.5))]*2.2, r"shower reliable",
       color="#1f77b4", fontsize=11, ha="center")
a.text(90, fo_scaled[np.argmin(np.abs(fo_ctr-90))]*2.6, r"FO reliable", color="k", fontsize=11, ha="center")
a.text(XM*1.05, a.get_ylim()[1]*0.3, r"matching seam", color="0.3", fontsize=10)
a.set_ylabel(r"$(1/\sigma)\,d\sigma/dp_T$"); a.legend(loc="lower left", fontsize=11)

# ---------- Panel 2: the reweighted result
a = ax[1]
a.plot(ctr, hpr, color="0.55", lw=2.0, ls="--", label=r"PS+LO prior")
a.plot(fo_ctr, fo_scaled, color="k", lw=2.0, ls=":", label=r"fixed order")
a.plot(ctr, hpo, color="#d62728", lw=2.6, label=r"reweighted (MaxEnt)")
a.axvline(XM, color="0.3", lw=1.2); a.axvspan(XM, XHI, color="#ffd24d", alpha=0.16)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(1, 300); a.grid(alpha=0.25, which="both")
a.set_title(r"2.\ Reweight: shape $=$ shower below the seam, $=$ FO above", fontsize=13, loc="left")
a.annotate(r"follows the shower", xy=(6, hpo[np.argmin(np.abs(ctr-6))]),
           xytext=(2, hpo[np.argmin(np.abs(ctr-6))]*0.25), color="#d62728", fontsize=11,
           arrowprops=dict(arrowstyle="->", color="#d62728"))
a.annotate(r"bent onto FO", xy=(90, hpo[np.argmin(np.abs(ctr-90))]),
           xytext=(120, hpo[np.argmin(np.abs(ctr-90))]*6), color="#d62728", fontsize=11,
           arrowprops=dict(arrowstyle="->", color="#d62728"))
a.set_ylabel(r"$(1/\sigma)\,d\sigma/dp_T$"); a.legend(loc="lower left", fontsize=11)

# ---------- Panel 3: reweighted / prior
a = ax[2]
ratio = hpo / np.maximum(hpr, 1e-30)
zinv = np.mean(ratio[(ctr < XM) & (hpr > 0)])
a.plot(ctr, ratio, color="#d62728", lw=2.6)
a.axhline(1.0, color="k", lw=1.0)
a.axhline(zinv, color="#1f77b4", lw=1.4, ls="--")
a.axvline(XM, color="0.3", lw=1.2); a.axvspan(XM, XHI, color="#ffd24d", alpha=0.16)
a.set_xscale("log"); a.set_xlim(1, 300); a.set_ylim(0.5, 2.2); a.grid(alpha=0.25, which="both")
a.set_title(r"3.\ reweighted\,/\,prior: flat below (shape kept), shaped above", fontsize=13, loc="left")
a.text(2.2, zinv+0.06, rf"FLAT at $1/Z={zinv:.2f}$" "\n" r"shape unchanged" "\n" r"height rescaled to the FO rate",
       color="#1f77b4", fontsize=10.5, va="bottom")
a.text(70, 1.55, r"shape reshaped to FO", color="#d62728", fontsize=11, ha="center")
a.set_ylabel("reweighted / prior"); a.set_xlabel(r"$p_T^{\ell\ell}$ [GeV]")

fig.suptitle("How the MaxEnt matching reweighting works  (DY, $p_T$ of the Z)", fontsize=15, y=0.995)
out = os.path.join(HERE, "fig_explain_matching.png")
fig.savefig(out); fig.savefig(out.replace(".png", ".pdf")); print("wrote", out)
print(f"effN={100*res.effN:.0f}%  below-seam 1/Z={zinv:.3f}  window-rate(FO)={M['recoil']['pT_ll']['rate']:.3f}")
