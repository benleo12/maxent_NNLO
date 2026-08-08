#!/usr/bin/env python3
r"""Solve-quality diagnostic for the unified upgrade(): pulls, gradient, matching.

Reconstructs the EXACT constraints _impose() builds (Chebyshev features C[:,n],
composite recoil target, Born full-range target), solves with the same z-scored
Newton (_newton_maxent, L2=1e-4), and reports for every imposed moment:

  target      mu_n         (what the solver imposes: composite for recoil)
  achieved    sum_i q_i C[:,n]_i
  residual    achieved - target      (= -L2*lam_n*sd_n at the optimum)
  sigma       FO moment uncertainty
  pull        residual / sigma       (matched within the FO error?)
  SNR         |mu - prior| / sigma   (is the constraint even informative?)

plus the dual gradient |grad| (must be ~0) and its max component.  This is the
'is the gradient zero and are the moments matching' check.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from maxent_upgrade.dy_method import cheb as _cheb, umap as _umap
from maxent_upgrade.maxent_match import _newton_maxent
from nnlojet_moments import fo_moments_from_nnlojet

BASE = "/Users/user/nnlojet-v1.0.2/dy_profile_poc"
RUN = "DY_MOMENTS"
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
SEEDS = [1, 2]
L2 = 1e-4
CFG = dict(born={"mll": {"range": (66., 116.), "map": "lin"},
                 "y_abs": {"range": (0., 2.4), "map": "lin"}},
           recoil={"pT_ll": {"range": (0.5, 500.), "map": "log", "soft_lo": 0.5}})


def main():
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v2.npz")))
    ev = dict(mll=P["mll"].astype(float), y_abs=np.abs(P["y_ll"]).astype(float),
              pT_ll=P["pT_ll"].astype(float), weight=P["w"].astype(float))
    n = len(ev["weight"])
    idx = np.random.default_rng(0).choice(n, min(400000, n), replace=False)
    ev = {k: v[idx] for k, v in ev.items()}
    p = ev["weight"] / ev["weight"].sum()

    M = fo_moments_from_nnlojet(BASE, RUN, CH6, SEEDS,
                                born_tags={"mll": "mll", "y_abs": "absyz"},
                                recoil_tag="ptz", recoil_cfg_name="pT_ll",
                                n_born=6, n_recoil=12, x_match=30., x_hi=500., soft_lo=0.5)

    F, mu, sig, snr, names = [], [], [], [], []

    def impose(obs, lo, hi, mp, XM, XHI, vals, errs, rate):
        XL = ev[obs]; Nmax = len(vals)
        win = (XL >= XM) & (XL < XHI)
        mu_prior_win = (np.array([(p[win] * _cheb(_umap(np.clip(XL[win], lo, hi), lo, hi, mp), Nmax)[:, k]).sum()
                                   / p[win].sum() for k in range(1, Nmax + 1)])
                        if win.any() else np.zeros(Nmax))
        C = _cheb(_umap(np.clip(XL, lo, hi), lo, hi, mp), max(Nmax, 1))
        I_S = XL < XM; I_T = XL >= XHI
        P_tail = float(p[I_T].sum()); P_soft = 1.0 - rate - P_tail; wS = float(p[I_S].sum())
        for k in range(1, Nmax + 1):
            soft = float((p[I_S] * C[I_S, k]).sum()) / wS if wS > 0 else 0.0
            tail = float((p[I_T] * C[I_T, k]).sum()) / P_tail if P_tail > 0 else 0.0
            F.append(C[:, k]); mu.append(P_soft * soft + rate * float(vals[k - 1]) + P_tail * tail)
            s = float(errs[k - 1]) if k - 1 < len(errs) else np.nan
            sig.append(s); names.append(f"{obs}_T{k}")
            snr.append(abs(float(vals[k - 1]) - mu_prior_win[k - 1]) / max(s, 1e-30))

    for o in CFG["born"]:
        a, b = CFG["born"][o]["range"]; mp = CFG["born"][o].get("map", "lin")
        impose(o, a, b, mp, -np.inf, np.inf, np.asarray(M["born"][o]["values"], float),
               np.asarray(M["born"][o]["errors"], float), rate=1.0)
    rc = M["recoil"]["pT_ll"]
    impose("pT_ll", float(rc["soft_lo"]), float(rc["x_hi"]), "log",
           float(rc["x_match"]), float(rc["x_hi"]),
           np.asarray(rc["window_values"], float), np.asarray(rc["window_errors"], float),
           rate=float(rc["rate"]))

    Phi = np.column_stack(F); mu = np.asarray(mu); sig = np.asarray(sig); snr = np.asarray(snr)
    q, lam, ok = _newton_maxent(Phi, p, mu, l2=L2)

    achieved = q @ Phi
    resid = achieved - mu
    # z-scored dual gradient (solver's convergence quantity)
    m_L = (p[:, None] * Phi).sum(0)
    sd = np.maximum(np.sqrt((p[:, None] * (Phi - m_L) ** 2).sum(0) + 1e-30), 1e-12)
    Phz = (Phi - m_L) / sd; muz = (mu - m_L) / sd
    grad = q @ Phz - muz + L2 * lam
    pull = resid / np.where(sig > 0, sig, np.nan)
    effN = 1.0 / (len(q) * float((q ** 2).sum()))

    print(f"solve: converged={ok}   |grad|_zscored={np.linalg.norm(grad):.2e}   "
          f"max|grad_i|={np.max(np.abs(grad)):.2e}   effN={100*effN:.1f}%   pos={(q>0).all()}")
    print(f"{'moment':>11} | {'target':>9} | {'achieved':>9} | {'resid':>9} | "
          f"{'sigma_FO':>9} | {'pull':>7} | {'SNR':>6}")
    print("-" * 78)
    for i, nm in enumerate(names):
        print(f"{nm:>11} | {mu[i]:>9.4f} | {achieved[i]:>9.4f} | {resid[i]:>+9.1e} | "
              f"{sig[i]:>9.1e} | {pull[i]:>+7.2f} | {snr[i]:>6.1f}")
    print("-" * 78)
    good = np.isfinite(pull)
    print(f"worst |resid/target| (closure) = {np.max(np.abs(resid/np.where(np.abs(mu)>1e-12,mu,1e-12))):.2e}")
    print(f"max |pull| (residual vs FO error) = {np.nanmax(np.abs(pull[good])):.3f}   "
          f"[<1 => every moment matched within its own FO uncertainty]")
    print(f"# constraints with SNR>1 (informative) = {(snr>1).sum()}/{len(snr)}")


if __name__ == "__main__":
    main()
