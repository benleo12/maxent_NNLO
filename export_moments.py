#!/usr/bin/env python3
r"""Export the fixed-order moments of the paper in one self-describing JSON per process.

    python export_moments.py            # all processes -> moments/<name>.json
    python export_moments.py dy w aa    # a subset

One schema for every process (read it back with moments/load_moments.py):

  process, order, sqrt_s_TeV, pdf, ew_scheme, scale, fiducial, n_independent_runs
  sigma_fid_pb        {value, stat, scale_lo, scale_hi}
  towers[key]         kind = "born" | "window" | "reference"
                      observable (the events-dict key the solver expects), map, range,
                      [bw = [M, Gamma] for the Breit-Wigner map],
                      n_orders, values[n], stat_errors[n], scale_errors[n], note,
                      window towers add: soft_lo, profile {a, b, c[, d]}, window_rate (+ _stat, _scale)
  joint[key]          observables [x, y], n, profile (or null), values {"m,n": ...},
                      stat_errors, scale_errors, window_rate (+ _stat, _scale)
  default_constraints the towers the paper imposed: {born: [...], recoil: [...], mixed: [...]}

Definitions (all on the fixed-order phase-space points, weight = dsigma):
  Born tower       mu_n = <T_n(u(v))>                       over the fiducial region
  window tower     mu_n = <w(x) T_n(u(x))> / <w(x)>,        window rate R = <w> / sigma_fid
  joint            mu_mn = <w_x w_y T_m(u(x)) T_n(u(y))> / <w_x w_y>, its own rate
  u_lin = 2 (v-lo)/(hi-lo) - 1,  u_log = 2 (ln v - ln lo)/(ln hi - ln lo) - 1,
  u_BW  = 2 (t(v)-t(lo))/(t(hi)-t(lo)) - 1,  t(v) = arctan((v^2-M^2)/(M Gamma)),
  w(x)  = 6t^5-15t^4+10t^3, t = clip((ln x - ln a)/(ln b - ln a), 0, 1), hard cut at c.
  values = pooled over the independent runs (ratio of summed numerators to summed denominators),
  stat = run-to-run scatter / sqrt(n_runs), scale = half-range over the seven-point variation.
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from nnlojet_moments import common_seeds, _moment_over_seeds, _reduce, _sum_channels  # noqa: E402

ROOT = os.environ.get("NNLOJET_ROOT", os.path.expanduser("~/nnlojet-v1.0.2"))
OUT = os.environ.get("MOMENTS_OUT", os.path.join(HERE, "moments_release"))
MZ, GZ = 91.1876, 2.4952
MW, GW = 80.385, 2.085
NNLO = ["LO", "R", "V", "RR", "RV", "VV"]

def T(key, kind, tag, den, mp, rng, obs, note, **kw):
    d = dict(key=key, kind=kind, tag=tag, den=den, map=mp, range=list(rng), observable=obs, note=note); d.update(kw); return d

# ------------------------------------------------------------------ the productions
SPEC = {
 "dy": dict(file="dy_nnlo_moments", base="dy_profile_log30_hi", run="DY_MOMENTS", prefix="Z", ch=NNLO,
    meta=dict(process="pp -> Z/gamma* -> l+ l-", order="NNLO QCD (NNLOJET)", sqrt_s_TeV=13, pdf="CT18NNLO",
              ew_scheme="G_mu", scale="muR = muF = m_ll",
              fiducial="66 < m_ll < 116 GeV, pT_l > 27 GeV, |eta_l| < 2.5, Born-level leptons (ATLAS 13 TeV)"),
    towers=[T("mll", "born", "prof_mll", "norm_born", "bw", (66.0, 116.0), "mll", "dilepton mass", bw=[MZ, GZ]),
            T("y_abs", "born", "prof_absyz", "norm_born", "lin", (0.0, 2.4), "y_abs", "|y_ll|"),
            T("pT_ll", "window", "prof_wptz", "prof_wptz_0", "log", (30.0, 2500.0), "pT_ll",
              "recoil, C^2 window 30-60 GeV, hard cut 500 GeV", soft_lo=30.0, profile=dict(a=30.0, b=60.0, c=500.0)),
            T("costh", "reference", "prof_ctl", "norm_born", "lin", (-1.0, 1.0), "costh",
              "Collins-Soper cos theta* of the negative lepton; booked, not imposed in the paper"),
            T("pT_l1", "reference", "prof_ptl1", "norm_born", "log", (27.0, 1000.0), "pT_lead",
              "leading-lepton pT; a prediction of the paper, released for validation, never imposed"),
            T("pT_l2", "reference", "prof_ptl2", "norm_born", "log", (27.0, 1000.0), "pT_sub",
              "subleading-lepton pT; a prediction of the paper, released for validation, never imposed")],
    joint=[dict(key="mll_x_costh", tag="prof_mct", den="norm_born", observables=["mll", "costh"], profile=None,
                note="<T_m(u_BW(m_ll)) T_n(cos theta*)>, no window; closes the decay correlation, redundant for the lepton spectra (paper)")],
    default=dict(born=["mll", "y_abs"], recoil=["pT_ll"], mixed=[])),
 "wp": dict(file="w_nnlo_moments_WP", base="w_moments/WP", run="W_MOMENTS", prefix="WP", ch=NNLO,
    meta=dict(process="pp -> W+ -> l+ nu", order="NNLO QCD (NNLOJET)", sqrt_s_TeV=7, pdf="CT18NNLO",
              ew_scheme="G_mu (complex masses), CKM full", scale="muR = muF = m_lnu",
              fiducial="pT_l > 20 GeV, |eta_l| < 2.5, pT_nu > 25 GeV, m_T > 40 GeV, Born-level leptons (ATLAS 7 TeV)")),
 "wm": dict(file="w_nnlo_moments_WM", base="w_moments/WM", run="W_MOMENTS", prefix="WM", ch=NNLO,
    meta=dict(process="pp -> W- -> l- nubar", order="NNLO QCD (NNLOJET)", sqrt_s_TeV=7, pdf="CT18NNLO",
              ew_scheme="G_mu (complex masses), CKM full", scale="muR = muF = m_lnu",
              fiducial="pT_l > 20 GeV, |eta_l| < 2.5, pT_nu > 25 GeV, m_T > 40 GeV, Born-level leptons (ATLAS 7 TeV)")),
 "aa": dict(file="aa_nnlo_moments", base="gg_moments2", run="GG_MOMENTS", prefix="GG", ch=NNLO,
    meta=dict(process="pp -> gamma gamma (continuum, q qbar + q g + g g channels)", order="NNLO QCD (NNLOJET)", sqrt_s_TeV=8,
              pdf="CT18NNLO", ew_scheme="alpha(0)", scale="muR = muF = m_gg",
              fiducial="pT_g1 > 40, pT_g2 > 30 GeV, |eta_g| < 2.37 excl. 1.37-1.56, dR_gg > 0.4, Frixione isolation r=0.4 ET_max=11 GeV (ATLAS 8 TeV)"),
    towers=[T("m_aa", "born", "prof_maa", "norm_born", "log", (80.0, 700.0), "m_aa", "diphoton mass"),
            T("costh_aa", "born", "prof_cts", "norm_born", "lin", (0.0, 1.0), "costh_aa", "|cos theta*| (Collins-Soper)"),
            T("pt_aa", "window", "prof_wpt", "prof_wpt_0", "log", (1.0, 500.0), "pt_aa",
              "recoil, C^2 window 28-56 GeV, hard cut 500 GeV", soft_lo=1.0, profile=dict(a=28.0, b=56.0, c=500.0))],
    joint=[], default=dict(born=["m_aa", "costh_aa"], recoil=["pt_aa"], mixed=[])),
 "ggh": dict(file="ggh_nnlo_moments", base="ggh_moments2", run="GGH_MOMENTS", prefix="H", ch=NNLO,
    meta=dict(process="pp -> H (gluon fusion, heavy-top effective theory)", order="NNLO QCD (NNLOJET)", sqrt_s_TeV=13,
              pdf="CT18NNLO", ew_scheme="-", scale="muR = muF = m_H", fiducial="none (fully inclusive)",
              sigma_fid_note="NOT VALIDATED: the norm_born accumulator of this production carries an overall normalization that does not reproduce the known inclusive gg -> H cross section (it comes out about 600 times too large); the paper uses only the shape moments, which are ratios and unaffected. sigma_fid_pb is therefore set to null; take the inclusive cross section from elsewhere to normalize an upgraded sample."),
    towers=[T("y_abs", "born", "prof_yh", "norm_born", "lin", (0.0, 4.0), "y_abs", "|y_H|"),
            T("pT_H", "window", "prof_wpt", "prof_wpt_0", "log", (1.0, 1000.0), "pT_H",
              "recoil, C^2 window 37-74 GeV, hard cut 1000 GeV", soft_lo=1.0, profile=dict(a=37.0, b=74.0, c=1000.0))],
    joint=[], default=dict(born=["y_abs"], recoil=["pT_H"], mixed=[])),
 "zj": dict(file="zj_nlo_moments", base="zj_moments2", run="ZJ_MOMENTS", prefix="ZJ", ch=["LO", "R", "V"],
    meta=dict(process="pp -> Z(-> l+ l-) + jet", order="NLO QCD (NNLOJET)", sqrt_s_TeV=13, pdf="CT18NNLO",
              ew_scheme="G_mu", scale="muR = muF = m_ll",
              fiducial="66 < m_ll < 116 GeV, pT_l > 27 GeV, |eta_l| < 2.5 (Born-level leptons); anti-kT R=0.4 jets pT > 20 GeV, |y| < 4.4, dR(jet, lepton) > 0.4 (ATLAS 13 TeV)"),
    towers=[T("mll", "born", "prof_mll", "norm_born", "bw", (66.0, 116.0), "mll", "dilepton mass", bw=[MZ, GZ]),
            T("y_abs", "born", "prof_absyz", "norm_born", "lin", (0.0, 2.4), "y_abs", "|y_ll|"),
            T("ptj1", "window", "prof_wj1", "prof_wj1_0", "log", (10.0, 1000.0), "ptj1",
              "leading-jet pT, C^2 window 30-60 GeV, hard cut 1000 GeV", soft_lo=10.0, profile=dict(a=30.0, b=60.0, c=1000.0)),
            T("ptj2", "window", "prof_wj2", "prof_wj2_0", "log", (10.0, 1000.0), "ptj2",
              "second-jet pT, same window", soft_lo=10.0, profile=dict(a=30.0, b=60.0, c=1000.0)),
            T("pimdphi", "window", "prof_wdphi", "prof_wdphi_0", "log", (0.01, float(np.pi)), "pimdphi",
              "lepton-pair acoplanarity pi - dphi_ll, C^2 window 0.05-0.20; the paper imposed 4 orders", soft_lo=0.01,
              profile=dict(a=0.05, b=0.20, c=4.0))],
    joint=[dict(key="ptj1_x_ptj2", tag="prof_wj12", den="prof_wj12_0", observables=["ptj1", "ptj2"],
                profile=dict(ptj1=dict(a=30.0, b=60.0, c=1000.0, d=1000.0), ptj2=dict(a=30.0, b=60.0, c=1000.0, d=1000.0)),
                note="<w1 w2 T_m(u_log(pT_j1)) T_n(u_log(pT_j2))> / <w1 w2>; closes the non-factorizing jet-pair correlation (paper)")],
    default=dict(born=["mll", "y_abs"], recoil=["ptj1", "ptj2"], mixed=["ptj1_x_ptj2"])),
}
for k in ("wp", "wm"):
    SPEC[k].update(towers=[
        T("m_w", "born", "prof_mw", "norm_born", "bw", (40.0, 1000.0), "m_w", "l-nu invariant mass (Born-level)", bw=[MW, GW]),
        T("y_abs", "born", "prof_absyw", "norm_born", "lin", (0.0, 4.0), "y_abs", "|y_W|"),
        T("pt_w", "window", "prof_wptw", "prof_wptw_0", "log", (25.0, 2500.0), "pt_w",
          "recoil, C^2 window 25-50 GeV, hard cut 500 GeV", soft_lo=25.0, profile=dict(a=25.0, b=50.0, c=500.0)),
        T("ctw_window", "window", "prof_wctw", "prof_wctw_0", "lin", (0.0, 2.0), "ctw1",
          "signed Collins-Soper cos theta* of the charged lepton, shifted by +1 (events key ctw1 = cos theta* + 1); "
          "window |cos theta*| < 0.5 with a C^2 ramp to zero at 0.7; the paper's optional decay-angle constraint",
          soft_lo=0.0, profile=dict(a=0.3, b=0.4, c=1.6, d=1.7)),
        T("ctw_full", "reference", "prof_ctw", "norm_born", "lin", (-1.0, 1.0), "ctw",
          "the same angle over its full range; booked, shown in the paper to distort m_T, not recommended")],
        joint=[dict(key="pt_w_x_ctw", tag="prof_ptwctw", den="prof_wptwctw_0", observables=["pt_w", "ctw1"],
                    profile=dict(pt_w=dict(a=25.0, b=50.0, c=500.0, d=500.0), ctw1=dict(a=0.3, b=0.4, c=1.6, d=1.7)),
                    note="<w T_m(u_log(pT_W)) T_n(cos theta*)> / <w>; does not remove the m_T residual (paper)")],
        default=dict(born=["m_w", "y_abs"], recoil=["pt_w"], mixed=[]))


def booked_orders(base, run, ch0, prefix, tag):
    pat = os.path.join(base, f"ch_{ch0}", f"{prefix}.{run}.{ch0}.{tag}_*.s*.dat")
    ns = set()
    for f in glob.glob(pat):
        m = re.search(re.escape(f".{tag}_") + r"(\d+)\.s\d+\.dat$", f)
        if m:
            ns.add(int(m.group(1)))
    return ns


def rate(base, run, ch, seeds, prefix, wtag):
    ps, W, N = [], None, None
    for s in seeds:
        w = _sum_channels(base, run, ch, wtag, s, prefix); n = _sum_channels(base, run, ch, "norm_born", s, prefix)
        if w is None or n is None:
            continue
        ps.append(w[0] / n[0]); W = w.copy() if W is None else W + w; N = n.copy() if N is None else N + n
    pooled = W / N
    return float(pooled[0]), float(np.std(ps, ddof=1) / np.sqrt(len(ps))), float(0.5 * (pooled[1:7].max() - pooled[1:7].min()))


def export(name):
    S = SPEC[name]; base = os.path.join(ROOT, S["base"]); run, prefix, ch = S["run"], S["prefix"], S["ch"]
    seeds = common_seeds(base, run, ch, prefix=prefix)
    out = dict(S["meta"]); out.update(n_independent_runs=len(seeds), channels=ch, towers={}, joint={}, default_constraints=S["default"])
    # fiducial cross section: norm_born is a one-bin histogram, sigma = value x width (fb -> pb)
    f0 = glob.glob(os.path.join(base, f"ch_{ch[0]}", f"{prefix}.{run}.{ch[0]}.norm_born.s*.dat"))[0]
    r0 = [l.split() for l in open(f0) if l.strip() and not l.startswith("#")][0]; width = float(r0[2]) - float(r0[0])
    ps, Ssum = [], None
    for s in seeds:
        n = _sum_channels(base, run, ch, "norm_born", s, prefix)
        if n is not None:
            ps.append(n[0] * width); Ssum = n.copy() * width if Ssum is None else Ssum + n * width
    Ssum = Ssum / len(ps)
    out["sigma_fid_pb"] = dict(value=float(Ssum[0] / 1e3), stat=float(np.std(ps, ddof=1) / np.sqrt(len(ps)) / 1e3),
                               scale_lo=float(Ssum[1:7].min() / 1e3), scale_hi=float(Ssum[1:7].max() / 1e3))
    if "sigma_fid_note" in out:
        out["sigma_fid_pb"] = None
    for t in S["towers"]:
        ns = booked_orders(base, run, ch[0], prefix, t["tag"]); ns.discard(0)
        if not ns:
            print(f"  {name}: tower {t['key']} ({t['tag']}) not booked, skipped"); continue
        orders = sorted(ns)
        v, st, sc = [], [], []
        for n in orders:
            c, s_, sc_, _ = _reduce(_moment_over_seeds(base, run, ch, seeds, f"{t['tag']}_{n}", t["den"], prefix))
            v.append(c); st.append(s_); sc.append(sc_)
        d = dict(kind=t["kind"], observable=t["observable"], map=t["map"], range=t["range"], n_orders=len(orders), orders=orders,
                 values=v, stat_errors=st, scale_errors=sc, note=t["note"])
        if "bw" in t:
            d["bw"] = t["bw"]
        if t["kind"] == "window":
            R, Rs, Rc = rate(base, run, ch, seeds, prefix, t["den"])
            d.update(soft_lo=t["soft_lo"], profile=t["profile"], window_rate=R, window_rate_stat=Rs, window_rate_scale=Rc)
        out["towers"][t["key"]] = d
    for j in S["joint"]:
        vals, st, sc = {}, {}, {}
        for m in range(1, 5):
            for n in range(1, 5):
                mm = _moment_over_seeds(base, run, ch, seeds, f"{j['tag']}_{m}{n}", j["den"], prefix)
                if mm[0] is None:
                    continue
                c, s_, sc_, _ = _reduce(mm); vals[f"{m},{n}"] = c; st[f"{m},{n}"] = s_; sc[f"{m},{n}"] = sc_
        if not vals:
            print(f"  {name}: joint {j['key']} not booked, skipped"); continue
        d = dict(observables=j["observables"], n=4, profile=j["profile"], values=vals, stat_errors=st, scale_errors=sc, note=j["note"])
        if j["den"] != "norm_born":
            R, Rs, Rc = rate(base, run, ch, seeds, prefix, j["den"]); d.update(window_rate=R, window_rate_stat=Rs, window_rate_scale=Rc)
        else:
            d.update(window_rate=1.0, window_rate_stat=0.0, window_rate_scale=0.0)
        out["joint"][j["key"]] = d
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, S["file"] + ".json"), "w") as f:
        json.dump(out, f, indent=1)
    rows = ["tower,kind,observable,map,lo,hi,order,value,stat,scale"]
    for k, d in out["towers"].items():
        for i, n in enumerate(d["orders"]):
            rows.append(f"{k},{d['kind']},{d['observable']},{d['map']},{d['range'][0]:g},{d['range'][1]:g},{n},{d['values'][i]:.8g},{d['stat_errors'][i]:.3g},{d['scale_errors'][i]:.3g}")
    for k, d in out["joint"].items():
        for key_, v in d["values"].items():
            rows.append(f"{k},joint,{'+'.join(d['observables'])},-,-,-,{key_.replace(',', '')},{v:.8g},{d['stat_errors'][key_]:.3g},{d['scale_errors'][key_]:.3g}")
    with open(os.path.join(OUT, S["file"] + ".csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    sig = out["sigma_fid_pb"] or dict(value=float("nan"), stat=float("nan"), scale_lo=float("nan"), scale_hi=float("nan"))
    print(f"{name:4s} {S['file']}: {len(seeds)} runs; sigma_fid {sig['value']:.2f} pb (stat {sig['stat']:.2f}, scale {sig['scale_lo']:.2f}-{sig['scale_hi']:.2f}); "
          f"towers {[(k, d['n_orders']) for k, d in out['towers'].items()]}; joint {[(k, len(d['values'])) for k, d in out['joint'].items()]}")


if __name__ == "__main__":
    names = sys.argv[1:] or list(SPEC)
    for nm in names:
        export("wp" if nm == "w+" else "wm" if nm == "w-" else nm)
