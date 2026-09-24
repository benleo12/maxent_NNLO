#!/usr/bin/env python3
r"""Transport through the prior's correlations (JHEP, transport section).

Left panel -- the first-order law, tested.  With l = ln(q/p) the log-weight of
the tilt, the shift of any observable y is, to first order in the multipliers,

    <y>_q - <y>_p  =  Cov_p(y, l)  +  O(l^2)

(Eq. transport, since l = sum_a lambda_a m_a - ln Z).  The first-order
prediction is plotted against the shift measured after the full nonlinear
solve, both in units of the prior standard deviation of y.  The residual IS the
higher-order transport; for the Drell-Yan tilt it is small but not zero, and
the script prints it.  (The all-orders identity <y>_q - <y>_p = Cov_p(q/p, y)
holds for ANY reweighting and tests nothing; the earlier version of this
figure plotted that.)

Right panel -- the pre-solve coupling.  Because l lies in the span of the
measurement functions, Cauchy-Schwarz gives

    |<y>_q - <y>_p| / sigma_y  <=  R_y * sigma_l  +  O(l^2),

with R_y the multiple correlation of y with the measurement functions under
the PRIOR -- computable before any fixed-order input exists -- and sigma_l the
prior standard deviation of the log-weight, one number for the whole solve.
Bars: the bound R_y*sigma_l.  Markers: the measured shift.  An observable with
R_y = 0 cannot be moved at first order by any choice of multipliers.
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
       ("pT_ll", r"$p_T^{\ell\ell}$", "C"), ("phistar", r"$\phi^*_\eta$", "F"),
       ("pT_lead", r"$p_T^{\ell,\rm lead}$", "F"),
       ("logpt", r"$\ln p_T^{\ell\ell}$", "D"), ("mll2", r"$m_{\ell\ell}^2$", "D")]
ev["logpt"] = np.log(np.maximum(ev["pT_ll"], 1e-6)); ev["mll2"] = ev["mll"]**2

pred, act, R, labs, kinds = [], [], [], [], []
for k, lab, kind in OBS:
    y = ev[k]; ybar = (p*y).sum()
    sd = np.sqrt(max((p*(y-ybar)**2).sum(), 1e-300))
    first = (p*(y-ybar)*(ell-ellbar)).sum()        # Cov_p(y, l): first order in the multipliers
    shift = (q*y).sum() - ybar                      # exact shift after the full solve
    pred.append(first/sd); act.append(shift/sd); R.append(multiple_corr(y))
    labs.append(lab); kinds.append(kind)
pred, act, R = np.array(pred), np.array(act), np.array(R)
col_of = lambda kd: C["maxent"] if kd == "C" else (C["minnlo"] if kd == "F" else C["mcatnlo"])

fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.2), gridspec_kw={"wspace": 0.42})
a = ax[0]
lim = 1.25*max(np.abs(pred).max(), np.abs(act).max(), 1e-6)
a.plot([-lim, lim], [-lim, lim], color="0.6", lw=1.6, ls="--")
_rank = {j: i for i, j in enumerate(np.argsort(-np.abs(pred)))}
for j, (pr, ac, lb, kd) in enumerate(zip(pred, act, labs, kinds)):
    a.plot(pr, ac, "o", ms=13, color=col_of(kd), zorder=5)
    up = _rank[j] % 2 == 0
    off = (-16, 14) if up else (16, -20)
    a.annotate(lb, (pr, ac), textcoords="offset points", xytext=off, fontsize=15,
               ha="right" if up else "left", color=col_of(kd), zorder=6)
_H = [plt.Line2D([], [], ls="", marker="o", ms=11, color=c_, label=l_) for c_, l_ in
      ((C["maxent"], r"constrained"), (C["minnlo"], r"predicted"),
       (C["mcatnlo"], r"derived from a constraint"))]
a.legend(handles=_H, loc="upper left", fontsize=14, labelspacing=0.35)
a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
a.set_xlabel(r"first order:  $\mathrm{Cov}_p\!\big(y,\ln(q/p)\big)/\sigma_y$")
a.set_ylabel(r"measured:  $(\langle y\rangle_q-\langle y\rangle_p)/\sigma_y$")

b = ax[1]
o = np.argsort(-R)
b.barh(range(len(o)), (R*sig_l)[o], color=[col_of(kinds[j]) for j in o], alpha=0.40,
       label=r"first-order bound $R_y\,\sigma_{\ln q/p}$")
b.plot(np.abs(act)[o], range(len(o)), "k|", ms=20, mew=2.6,
       label=r"measured $|\langle y\rangle_q-\langle y\rangle_p|/\sigma_y$")
b.set_yticks(range(len(o))); b.set_yticklabels([labs[j] for j in o])
b.invert_yaxis(); b.set_xlabel(r"shift in units of $\sigma_y$")
b.set_xlim(0, None)
b.legend(loc="lower right", fontsize=13)
fig.savefig(os.path.join(HERE, "fig_covariance_law.pdf"))
fig.savefig(os.path.join(HERE, "fig_covariance_law.png"))
rel = np.abs(pred-act)/np.maximum(np.abs(act), 1e-12)
print(f"wrote fig_covariance_law   sigma_l = {sig_l:.4f}   effN = {100*res.effN:.1f}%   "
      f"features kept {int(keep.sum())}/{Phi.shape[1]}")
print(f"  first-order vs measured: max |diff| = {np.max(np.abs(pred-act)):.2e} sigma_y;  "
      f"relative: max {100*rel.max():.1f}%, median {100*np.median(rel):.1f}%")
for pr, ac, r_, lb in zip(pred, act, R, labs):
    print(f"   {lb:22s} first-order {pr:+.5f}   measured {ac:+.5f}   R_y {r_:.3f}   bound {r_*sig_l:.4f}")
