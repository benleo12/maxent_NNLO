#!/usr/bin/env python3
r"""Export the NNLO Drell-Yan moments in a plain, self-describing form.

Writes dy_nnlo_moments.json and dy_nnlo_moments.csv: for each constrained tower,
the Chebyshev order, the central value, its statistical error (scatter over the
independent fixed-order runs) and its scale error (half-range over the 7-point
variation), plus the map and window each tower is defined on.

Definition, so the numbers can be reproduced or re-derived elsewhere:
  Born towers    mu_n = <T_n(u(v))>   averaged over fiducial events, weight = dsigma
  recoil tower   mu_n = <w(pT) T_n(u(pT))> / <w(pT)>, w the C^2 window below,
                 with the window rate R = <w(pT)> / sigma_fid quoted separately.
  u_lin(v)  = 2 (v - lo)/(hi - lo) - 1
  u_log(v)  = 2 (ln v - ln lo)/(ln hi - ln lo) - 1
  u_BW(v)   = 2 (t(v) - t(lo))/(t(hi) - t(lo)) - 1,  t(v) = arctan((v^2 - mZ^2)/(mZ GammaZ))
  w(pT)     = 6t^5 - 15t^4 + 10t^3,  t = clip(ln(pT/vm)/ln 2, 0, 1),  vm = 30 GeV,
              with a hard upper cut at 500 GeV.
Fiducial selection: 66 < m_ll < 116 GeV, pT_l > 27 GeV, |eta_l| < 2.5, Born leptons,
13 TeV, CT18NNLO, G_mu scheme, mu_R = mu_F = m_ll.
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from nnlojet_moments import fo_moments_smooth_from_nnlojet, common_seeds
BASE = os.path.expanduser("~/nnlojet-v1.0.2/dy_profile_log30_hi")
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
KW = dict(born_tags={"mll": "mll", "y_abs": "absyz"}, n_born=12, n_recoil=20,
          x_match=30.0, x_hi=2500.0, soft_lo=30.0)
MAPS = {"mll": ("BW", 66.0, 116.0), "y_abs": ("lin", 0.0, 2.4), "pT_ll": ("log", 30.0, 2500.0)}


def main():
    seeds = common_seeds(BASE, "DY_MOMENTS", CH6)
    M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, seeds, **KW)
    out = {"process": "pp -> Z -> l+l-, NNLO in QCD (NNLOJET)", "sqrt_s_TeV": 13,
           "pdf": "CT18NNLO", "ew_scheme": "G_mu", "scale": "muR = muF = m_ll",
           "fiducial": "66 < m_ll < 116 GeV, pT_l > 27 GeV, |eta_l| < 2.5, Born leptons",
           "n_independent_runs": len(seeds), "towers": {}}
    rows = [("tower", "map", "lo", "hi", "order", "value", "stat", "scale")]
    for grp in ("born", "recoil"):
        for obs, d in (M.get(grp) or {}).items():
            vk = "values" if grp == "born" else "window_values"
            v = np.asarray(d[vk], float)
            st = np.asarray(d.get("stat_errors", []), float)
            sc = np.asarray(d.get("scale_errors", []), float)
            mp, lo, hi = MAPS[obs]
            t = {"kind": grp, "map": mp, "range": [lo, hi], "n_orders": len(v),
                 "values": v.tolist(), "stat_errors": st.tolist(), "scale_errors": sc.tolist()}
            if grp == "recoil":
                t["window"] = {"v_match_GeV": 30.0, "v_full_GeV": 60.0, "v_cut_GeV": 500.0,
                               "profile": "6t^5-15t^4+10t^3, t = clip(ln(pT/30)/ln2,0,1)"}
                t["window_rate"] = float(d["rate"])
                t["window_rate_stat"] = float(d.get("rate_stat", float("nan")))
                t["window_rate_scale"] = float(d.get("rate_scale", float("nan")))
            out["towers"][obs] = t
            for n in range(len(v)):
                rows.append((obs, mp, lo, hi, n + 1, v[n],
                             st[n] if n < len(st) else float("nan"),
                             sc[n] if n < len(sc) else float("nan")))
    with open(os.path.join(HERE, "dy_nnlo_moments.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(HERE, "dy_nnlo_moments.csv"), "w") as f:
        for r in rows:
            f.write(",".join(f"{x:.8g}" if isinstance(x, float) else str(x) for x in r) + "\n")
    print(f"wrote dy_nnlo_moments.json and .csv from {len(seeds)} independent runs")
    for obs, t in out["towers"].items():
        v = np.asarray(t["values"]); st = np.asarray(t["stat_errors"]); sc = np.asarray(t["scale_errors"])
        print(f"  {obs:7s} {t['map']:3s} [{t['range'][0]:g},{t['range'][1]:g}] n={t['n_orders']:2d}  "
              f"median stat {np.median(st):.2e}  median scale {np.median(sc):.2e}"
              + (f"  rate {t['window_rate']:.4f}" if "window_rate" in t else ""))


if __name__ == "__main__":
    main()
