#!/usr/bin/env python3
r"""JHEP figure: the transport law on the Drell-Yan NNLO upgrade (fig_transport_law.pdf).

With l = ln(q/p) the log-weight of the tilt, the shift of the mean of any
observable v is, to first order in the multipliers,

    <v>_q - <v>_p  =  Cov_p(v, l)  +  O(l^2),

and because l lies in the span of the measurement functions, Cauchy-Schwarz
bounds that first-order shift by

    |Cov_p(v, l)| / sigma_v  <=  R_v * sigma_l,

with R_v the multiple correlation of v with the measurement functions under the
PRIOR (a regression on the unweighted sample) and sigma_l the prior standard
deviation of the log-weight, one number per solve.

One row per observable: the three constrained ones (m_ll, |y_ll|, pT_ll) and
two followers (the leading-lepton pT and phi*_eta).  Bar = the bound
R_v sigma_l; open circle = the first-order shift; black tick = the shift of the
mean measured after the full non-linear solve, all in units of the prior
standard deviation sigma_v.  Inputs: the 298-run dy_profile_log30_hi production
and the Born-level prior dy_prior_atlas_v3.npz (1M-event subsample, rng 0), with
the paper's constraint set (12 BW m_ll, 12 lin |y_ll|, 20 windowed pT_ll).
Same computation as fig_covariance_law.py (2026-09-27), redrawn as one panel.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pubstyle import use_pub_style, C
use_pub_style(base=18)
from maxent_upgrade import upgrade
from nnlojet_moments import fo_moments_smooth_from_nnlojet, common_seeds

BASE = os.path.join(os.environ.get("NNLOJET_ROOT",
        os.path.expanduser("~/nnlojet-v1.0.2")), "dy_profile_log30_hi")
CH6 = ["LO","R","V","RR","RV","VV"]; XM, XHI, SOFT, XMAP = 30.0, 500.0, 30.0, 2500.0

P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v3.npz")))
i = np.random.default_rng(0).choice(len(P["w"]), min(1_000_000, len(P["w"])), replace=False)
ev = dict(mll=P["mll"][i].astype(float), y_abs=np.abs(P["y_ll"][i]).astype(float),
          pT_ll=P["pT_ll"][i].astype(float), phistar=P["phistar"][i].astype(float),
          pT_lead=P["pT_lead"][i].astype(float), weight=P["w"][i].astype(float))
M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6,
        common_seeds(BASE, "DY_MOMENTS", CH6), born_tags={"mll":"mll","y_abs":"absyz"},
        n_born=12, n_recoil=20, x_match=XM, x_hi=XMAP, soft_lo=SOFT)
res = upgrade(ev, M, dict(
    born={"mll":{"range":(66.,116.),"map":"bw"},"y_abs":{"range":(0.,2.4),"map":"lin"}},
    recoil={"pT_ll":{"range":(SOFT,XMAP),"map":"log","soft_lo":SOFT,
                     "profile":{"a":XM,"b":2*XM,"c":XHI}}},
    followers=["phistar","pT_lead"], keep_features=True))
p = ev["weight"]/ev["weight"].sum(); q = res.weights/res.weights.sum()
ell = np.log(q/np.maximum(p, 1e-300))                       # log-weight of the tilt
ellbar = (p*ell).sum(); sig_l = float(np.sqrt((p*(ell-ellbar)**2).sum()))

# Multiple correlation of y with the measurement functions under the prior:
# weighted least squares of y on the z-scored feature matrix (tiny ridge).
Phi = np.asarray(res.report["Phi"], float)
c = (p[:, None]*Phi).sum(0); s = np.sqrt((p[:, None]*(Phi-c)**2).sum(0))
keep = s > 1e-12*max(float(s.max()), 1e-300)
X = (Phi[:, keep]-c[keep])/s[keep]
G = X.T @ (p[:, None]*X); G = G + 1e-8*np.trace(G)/len(G)*np.eye(len(G))

def multiple_corr(y):
    ybar = (p*y).sum(); sd = np.sqrt(max((p*(y-ybar)**2).sum(), 1e-300))
    b = np.linalg.solve(G, X.T @ (p*(y-ybar)))
    yhat = X @ b
    return float(np.clip(np.sqrt((p*yhat**2).sum())/sd, 0.0, 1.0))

OBS = [("mll", r"$m_{\ell\ell}$", "C"), ("y_abs", r"$|y_{\ell\ell}|$", "C"),
       ("pT_ll", r"$p_\mathrm{T}^{\ell\ell}$", "C"),
       ("pT_lead", r"$p_\mathrm{T}^{\ell_1}$", "F"), ("phistar", r"$\phi^*_\eta$", "F")]

pred, act, R, labs, kinds = [], [], [], [], []
for k, lab, kind in OBS:
    y = ev[k]; ybar = (p*y).sum()
    sd = np.sqrt(max((p*(y-ybar)**2).sum(), 1e-300))
    first = (p*(y-ybar)*(ell-ellbar)).sum()        # Cov_p(y, l): first order in the multipliers
    shift = (q*y).sum() - ybar                      # exact shift after the full solve
    pred.append(first/sd); act.append(shift/sd); R.append(multiple_corr(y))
    labs.append(lab); kinds.append(kind)
pred, act, R = np.array(pred), np.array(act), np.array(R)

# JHEP print size: set at 0.6\textwidth; pubstyle default sizes, nothing smaller.
fig, b = plt.subplots(figsize=(6.4, 4.3))
rows = np.arange(len(OBS))
bar_col = [C["prior"] if kd == "C" else C["maxent"] for kd in kinds]
b.barh(rows, R*sig_l, height=0.62, color=bar_col, alpha=0.35, lw=0, zorder=1)
b.plot(np.abs(pred), rows, "o", mfc="white", mec="k", mew=1.6, ms=9, zorder=3,
       label=r"first order, $|\mathrm{Cov}_p(v,\ln q/p)|/\sigma_v$")
b.plot(np.abs(act), rows, "k|", ms=22, mew=2.8, zorder=4,
       label=r"after the solve, $|\langle v\rangle_q-\langle v\rangle_p|/\sigma_v$")
b.barh([np.nan], [np.nan], color="0.6", alpha=0.5, lw=0,
       label=r"bound from the prior, $R_v\,\sigma_{\ln q/p}$")
b.set_yticks(rows); b.set_yticklabels(labs)
for t, kd in zip(b.get_yticklabels(), kinds):
    t.set_color("k" if kd == "C" else C["maxent"])
b.tick_params(axis="y", which="minor", left=False, right=False)
b.axhline(2.5, color="0.6", lw=0.8, ls="-")
b.invert_yaxis()
b.set_xlim(0, 0.20); b.set_ylim(len(OBS) - 0.45, -0.55)
b.text(0.197, 1.0, r"constrained", ha="right", va="center", color="0.3")
b.text(0.197, 3.5, r"followers", ha="right", va="center", color=C["maxent"])
b.set_xlabel(r"shift of the mean in units of $\sigma_v$")
b.legend(loc="lower center", bbox_to_anchor=(0.45, 1.01), ncol=1, labelspacing=0.25,
         handletextpad=0.5)
out = os.path.join(HERE, "fig_transport_law.pdf")
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
rel = np.abs(pred-act)/np.maximum(np.abs(act), 1e-12)
print(f"wrote fig_transport_law   sigma_l = {sig_l:.4f}   effN = {100*res.effN:.1f}%   "
      f"features kept {int(keep.sum())}/{Phi.shape[1]}")
print(f"  first-order vs measured: max |diff| = {np.max(np.abs(pred-act)):.2e} sigma_y;  "
      f"relative: max {100*rel.max():.1f}%, median {100*np.median(rel):.1f}%")
for pr, ac, r_, lb in zip(pred, act, R, labs):
    print(f"   {lb:28s} first-order {pr:+.5f}   measured {ac:+.5f}   R_v {r_:.3f}   bound {r_*sig_l:.4f}"
          f"   inside bound: {abs(ac) <= r_*sig_l}")
