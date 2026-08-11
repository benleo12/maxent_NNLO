#!/usr/bin/env python3
r"""Which observables move under reweighting, and by how much.

For any observable y, with r = q/p the per-event weight ratio,

    <y>_q - <y>_p  =  Cov_p(r, y) / <r>_p        (exact)

so an observable shifts if and only if it covaries with the tilt: zero
covariance implies exactly zero shift.  The left panel verifies the identity on
every observable we track; the right panel shows that the size of the shift is
governed by the covariance, NOT by whether the observable was constrained.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pubstyle import use_pub_style
use_pub_style(base=18)
from maxent_upgrade import upgrade
from nnlojet_moments import fo_moments_smooth_from_nnlojet, common_seeds

BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_poc"
CH6 = ["LO","R","V","RR","RV","VV"]; XM, XHI, SOFT = 30.0, 500.0, 0.5

P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v2.npz")))
i = np.random.default_rng(0).choice(len(P["w"]), 1_000_000, replace=False)
ev = dict(mll=P["mll"][i].astype(float), y_abs=np.abs(P["y_ll"][i]).astype(float),
          pT_ll=P["pT_ll"][i].astype(float), phistar=P["phistar"][i].astype(float),
          pT_lead=P["pT_lead"][i].astype(float), weight=P["w"][i].astype(float))
M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6,
        common_seeds(BASE, "DY_MOMENTS", CH6), born_tags={"mll":"mll","y_abs":"absyz"},
        n_born=6, n_recoil=12, x_match=XM, x_hi=XHI, soft_lo=SOFT)
res = upgrade(ev, M, dict(
    born={"mll":{"range":(66.,116.),"map":"lin"},"y_abs":{"range":(0.,2.4),"map":"lin"}},
    recoil={"pT_ll":{"range":(SOFT,XHI),"map":"log","soft_lo":SOFT,
                     "profile":{"a":XM,"b":2*XM,"c":XHI}}},
    followers=["phistar","pT_lead"]))
p = ev["weight"]/ev["weight"].sum(); q = res.weights/res.weights.sum()
r = q/np.maximum(p, 1e-300)                      # weight ratio
rbar = (p*r).sum()

OBS = [("mll", r"$m_{\ell\ell}$", "C"), ("y_abs", r"$|y_{\ell\ell}|$", "C"),
       ("pT_ll", r"$p_T^{\ell\ell}$", "C"), ("phistar", r"$\phi^*_\eta$", "F"),
       ("pT_lead", r"$p_T^{\ell,\rm lead}$", "F"),
       ("logpt", r"$\ln p_T^{\ell\ell}$", "D"), ("mll2", r"$m_{\ell\ell}^2$", "D")]
ev["logpt"] = np.log(np.maximum(ev["pT_ll"], 1e-6)); ev["mll2"] = ev["mll"]**2

pred, act, labs, kinds = [], [], [], []
for k, lab, kind in OBS:
    y = ev[k]; ybar = (p*y).sum()
    sd = np.sqrt(max((p*(y-ybar)**2).sum(), 1e-300))
    cov = (p*(r-rbar)*(y-ybar)).sum()/rbar
    shift = (q*y).sum() - ybar
    pred.append(cov/sd); act.append(shift/sd); labs.append(lab); kinds.append(kind)
pred, act = np.array(pred), np.array(act)

fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.2))
a = ax[0]
lim = 1.25*max(np.abs(pred).max(), np.abs(act).max(), 1e-6)
a.plot([-lim, lim], [-lim, lim], color="0.6", lw=1.6, ls="--")
# every point sits on the diagonal, so labels stacked on one side collide:
# offset them PERPENDICULAR to the diagonal, alternating side by rank.
_rank = {j: i for i, j in enumerate(np.argsort(-np.abs(pred)))}
for j, (pr, ac, lb, kd) in enumerate(zip(pred, act, labs, kinds)):
    col = "#d62728" if kd == "C" else ("#1f77b4" if kd == "F" else "#2ca02c")
    a.plot(pr, ac, "o", ms=13, color=col, zorder=5)
    up = _rank[j] % 2 == 0
    off = (-16, 14) if up else (16, -20)
    a.annotate(lb, (pr, ac), textcoords="offset points", xytext=off, fontsize=15,
               ha="right" if up else "left", color=col, zorder=6)
_H = [plt.Line2D([], [], ls="", marker="o", ms=11, color=c, label=l) for c, l in
      (("#d62728", r"constrained"), ("#1f77b4", r"predicted"),
       ("#2ca02c", r"derived from a constraint"))]
a.legend(handles=_H, loc="upper left", fontsize=14, labelspacing=0.35)
a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
a.set_xlabel(r"predicted  $\mathrm{Cov}_p(r,y)/\langle r\rangle\,/\,\sigma_y$")
a.set_ylabel(r"measured  $(\langle y\rangle_q-\langle y\rangle_p)/\sigma_y$")
a.set_title(r"the shift is exactly the covariance with the tilt")

b = ax[1]
o = np.argsort(-np.abs(pred))
cols = ["#d62728" if kinds[j]=="C" else ("#1f77b4" if kinds[j]=="F" else "#2ca02c") for j in o]
b.barh(range(len(o)), np.abs(pred)[o], color=cols)
b.set_yticks(range(len(o))); b.set_yticklabels([labs[j] for j in o])
b.invert_yaxis(); b.set_xlabel(r"$|\mathrm{Cov}_p(r,y)|/\langle r\rangle\,/\,\sigma_y$")
b.set_title(r"how strongly each observable is coupled to the tilt")
fig.savefig(os.path.join(HERE,"fig_covariance_law.pdf"))
fig.savefig(os.path.join(HERE,"fig_covariance_law.png"))
err = np.max(np.abs(pred-act))
print(f"wrote fig_covariance_law   max|predicted-measured| = {err:.2e} (in units of sigma_y)")
for pr, ac, lb in zip(pred, act, labs):
    print(f"   {lb:22s} predicted {pr:+.5f}   measured {ac:+.5f}   diff {abs(pr-ac):.2e}")
