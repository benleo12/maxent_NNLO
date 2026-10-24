#!/usr/bin/env python3
"""POSITIVE cross-moment demo: the Z+jet jet pair (ptj1, ptj2) is dynamically
correlated, so, unlike the DY decay pair, the marginals should NOT fix the
joint moments.  Measure the 16 windowed joint moments <w1 w2 T_m(u1) T_n(u2)>
under (prior | marginals-only solve | +16 mixed solve) against fixed order."""
import os, sys
os.environ["ZJ_CH"] = "LO,R,V"

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zj_mixed_scan as Z          # constants + prof_w + cheb_cols, main() guarded
from maxent_upgrade import upgrade
from nnlojet_moments import (fo_moments_smooth_from_nnlojet, common_seeds,
                             add_profiled_recoil, add_mixed_moments)

NMIX = 4
seeds = common_seeds(Z.ZDIR, Z.RUN, Z.CH,
                     tag=["prof_wj12_0"] + [f"prof_wj12_{m}{n}"
                                           for m in range(1, NMIX + 1)
                                           for n in range(1, NMIX + 1)],
                     prefix=Z.PREFIX)
print(f"seeds with joint moments: {len(seeds)}")

z = np.load(os.path.join(HERE, "zj_prior.npz"))
sel = (np.asarray(z["njet"]) >= 1) & np.isfinite(z["weight"]) & (z["weight"] > 0)
ev = {k: np.asarray(z[k])[sel] for k in ("mll", "y_abs", "ptj1", "ptj2", "pimdphi")}
ev["weight"] = np.asarray(z["weight"])[sel]

def build_M(with_mixed):
    M = fo_moments_smooth_from_nnlojet(
        Z.ZDIR, Z.RUN, Z.CH, seeds, born_tags={"mll": "mll", "y_abs": "absyz"},
        n_born=6, n_recoil=12, x_match=Z.XM, x_hi=Z.XHI, soft_lo=Z.SOFT,
        recoil_cfg_name="ptj1", norm_born="norm_born",
        w0="prof_wj1_0", wtag="prof_wj1", prefix=Z.PREFIX)
    add_profiled_recoil(M, Z.ZDIR, Z.RUN, Z.CH, seeds, "ptj2", wtag="prof_wj2",
                        w0="prof_wj2_0", n_recoil=6, x_match=Z.XM, x_hi=Z.XHI,
                        soft_lo=Z.SOFT, prefix=Z.PREFIX)
    add_mixed_moments(M, Z.ZDIR, Z.RUN, Z.CH, seeds, "ptj12", wtag="prof_wj12",
                      w0="prof_wj12_0", n_max=NMIX, prefix=Z.PREFIX)
    return M

M = build_M(True)
MX = M["mixed"]["ptj12"]
print("mixed keys:", {k: (type(v).__name__) for k, v in MX.items()})

recoil = {"ptj1": {"range": (Z.SOFT, Z.XHI), "map": "log", "soft_lo": Z.SOFT,
                   "profile": {"a": Z.XM, "b": Z.XB, "c": Z.XHI}},
          "ptj2": {"range": (Z.SOFT, Z.XHI), "map": "log", "soft_lo": Z.SOFT,
                   "profile": {"a": Z.XM, "b": Z.XB, "c": Z.XHI}}}
born = {"mll": {"range": (66., 116.), "map": "bw"},
        "y_abs": {"range": (0., 2.4), "map": "lin"}}
mixed_cfg = {"ptj12": dict(observables=("ptj1", "ptj2"),
                           range={"ptj1": (Z.SOFT, Z.XHI), "ptj2": (Z.SOFT, Z.XHI)},
                           map={"ptj1": "log", "ptj2": "log"},
                           profile={"ptj1": dict(a=Z.XM, b=Z.XB, c=Z.XHI, d=Z.XHI),
                                    "ptj2": dict(a=Z.XM, b=Z.XB, c=Z.XHI, d=Z.XHI)},
                           n=NMIX)}

rA = upgrade(ev, M, dict(born=born, recoil=recoil, mixed={}, followers=[],
                         moment_selection=False))
rB = upgrade(ev, M, dict(born=born, recoil=recoil, mixed=mixed_cfg, followers=[],
                         moment_selection=False, keep_features=True))
# absolute closure over ALL constraints of the full (marginals + joint) solve
_Phi = rB.report["Phi"]; _q = rB.weights / rB.weights.sum()
_ach = (_q[:, None] * _Phi).sum(0); _mu = rB.report["mu"]
_gap = np.abs(_ach - _mu)
print(f"full solve: {len(_mu)} constraints, max |achieved - target| = {_gap.max():.2e} "
      f"(feature {rB.report['feature_names'][int(np.argmax(_gap))]}), relative closure {rB.closure:.1e}")
# sigma_FO of the joint moments from the seed scatter (zj_cross_err.py), if available
_errf = os.path.join(HERE, "zj_cross_err.npz")
if os.path.exists(_errf):
    _E = np.load(_errf)
    errs_seed = {(int(m), int(n)): float(sg) for (m, n), sg in zip(_E["mn"], _E["sigma"])}
else:
    errs_seed = {}
print(f"A marginals-only: effN {100*rA.effN:.2f}%  closure {rA.closure:.1e}")
print(f"B +{NMIX*NMIX} mixed:    effN {100*rB.effN:.2f}%  closure {rB.closure:.1e}")

# measure joint moments exactly as upgrade() builds the features
w1 = Z.prof_w(ev["ptj1"], Z.XM, Z.XB, Z.XHI)
w2 = Z.prof_w(ev["ptj2"], Z.XM, Z.XB, Z.XHI)
wxy = w1 * w2
C1 = Z.cheb_cols(ev["ptj1"], Z.SOFT, Z.XHI, NMIX)
C2 = Z.cheb_cols(ev["ptj2"], Z.SOFT, Z.XHI, NMIX)

def joint(q, m, n):
    tot = q.sum(); R = float((q * wxy).sum() / tot)
    return float((q * wxy * C1[m] * C2[n]).sum() / tot) / max(R, 1e-300)

vals = MX["values"]; errs = MX.get("errors") or {}
errs = {**errs, **errs_seed} if errs_seed else errs
print("\n(m,n)   FO         +-err      prior      marginals  +mixed")
gaps = {"prior": [], "A": [], "B": []}
for (m, n), v in sorted(vals.items()):
    e_ = errs.get((m, n), np.nan)
    jp = joint(ev["weight"], m, n); ja = joint(rA.weights, m, n); jb = joint(rB.weights, m, n)
    gaps["prior"].append(abs(jp - v)); gaps["A"].append(abs(ja - v)); gaps["B"].append(abs(jb - v))
    print(f"({m},{n})  {v:+.4f}  {e_:9.4f}  {jp:+.4f}   {ja:+.4f}   {jb:+.4f}")
for k, g in gaps.items():
    print(f"mean |gap| {k:7s} {np.mean(g):.4f}   max {np.max(g):.4f}")
# machine-readable table for fig_zj_cross.py: (m,n) -> FO, sigma_FO, prior, marginals, +mixed
keys = sorted(vals)
np.savez(os.path.join(HERE, "zj_cross_results.npz"),
         mn=np.array(keys), fo=np.array([vals[k] for k in keys]),
         err=np.array([errs.get(k, np.nan) for k in keys]),
         prior=np.array([joint(ev["weight"], m, n) for m, n in keys]),
         marg=np.array([joint(rA.weights, m, n) for m, n in keys]),
         mixed=np.array([joint(rB.weights, m, n) for m, n in keys]),
         effN_marg=rA.effN, effN_mixed=rB.effN, n_events=len(ev["weight"]))
print("saved zj_cross_results.npz")
