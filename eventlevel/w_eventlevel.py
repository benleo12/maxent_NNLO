#!/usr/bin/env python3
r"""W production at NNLO with event-level moments: the color-singlet
construction of the Drell-Yan section with a W in place of the Z.

Constrained (per charge, W+ and W- separately)
  Born   : m_W    Breit-Wigner map [40,1000] around (M_W, Gamma_W)   n <= 12
           |y_W|  linear map [0,4]                                   n <= 12
  recoil : pT_W   log map [25,2500], profile [25,50] -> 500          n <= 20
The Born variables of the W are not measurable event by event -- the neutrino
carries no rapidity information -- which is the point of the exercise: the
constraints act on the unobservable Born kinematics, and every measured
quantity (lepton |eta|, pT_l with its Jacobian peak, m_T, ET_miss, the charge
asymmetry) is a follower predicted through the prior's decay correlations.

Fiducial region (ATLAS 7 TeV, 1109.5141 / Rivet ATLAS_2011_I928289_W):
  pT_l > 20, |eta_l| < 2.5, pT_nu > 25, m_T > 40 GeV; 7 TeV, CT18NNLO.
Data: dressed-muon tables d05 (W+), d03 (W-), d07 (asymmetry), y04 columns.

Environment knobs: NNLOJET_ROOT (default ~/nnlojet-v1.0.2), W_PRIOR_DIR
(default ~/w_workdir), W_CH (comma list of channels, default all six),
W_CHARGES (default WP,WM), W_NMAX (cap prior events per charge).
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import use_pub_style, C, LS, LW           # noqa: E402
use_pub_style(base=17)
from nnlojet_moments import (fo_moments_smooth_from_nnlojet, common_seeds,   # noqa: E402
                             fo_curve_band, _load, add_profiled_recoil, add_mixed_moments)


def load_moments(charge, seeds, scale_idx=0):
    """Central/variant moments for one charge, with the windowed decay-angle
    tower attached when W_CTW_MODE=window."""
    M = fo_moments_smooth_from_nnlojet(wdir(charge), RUN, CH6, seeds, scale_idx=scale_idx, prefix=charge, **LOADER_KW)
    if USE_CTW_WIN:
        add_profiled_recoil(M, wdir(charge), RUN, CH6, seeds, "ctw1", wtag="prof_wctw", w0="prof_wctw_0",
                            n_recoil=CTW_WIN_N, x_match=CTW_WIN["a"], x_hi=2.0, soft_lo=0.0,
                            scale_idx=scale_idx, prefix=charge)
    if USE_CTW_JOINT:
        add_mixed_moments(M, wdir(charge), RUN, CH6, seeds, "ptwctw", wtag="prof_ptwctw", w0="prof_wptwctw_0",
                          n_max=CTW_JOINT_N, scale_idx=scale_idx, prefix=charge)
    return M
from maxent_upgrade import upgrade, check_seam, profile_w                     # noqa: E402
from bandviz import stagger                                                   # noqa: E402
from yoda_ref import read as yoda_read                                        # noqa: E402

WROOT = os.path.join(os.environ.get("NNLOJET_ROOT", os.path.expanduser("~/nnlojet-v1.0.2")), "w_moments")
RUN = "W_MOMENTS"
CH6 = [c for c in os.environ.get("W_CH", "LO,R,V,RR,RV,VV").split(",") if c]
CHJ = tuple(c for c in ("R", "RR", "RV") if c in CH6) or tuple(CH6)   # channels with a recoil
CHARGES = [c for c in os.environ.get("W_CHARGES", "WP,WM").split(",") if c]
PRIOR_DIR = os.path.expanduser(os.environ.get("W_PRIOR_DIR", "~/w_workdir"))
# mirrors eval_w_ptw / eval_chebTw_ptw (pa=25, pb=50, pc=pd=500; log map [25,2500])
XM, XB, XHI, SOFT, XMAP = 25.0, 50.0, 500.0, 25.0, 2500.0
MW, GW = 80.385, 2.085
MLO, MHI, YHI = 40.0, 1000.0, 4.0
N_BOOT = 20
YODA = "/opt/homebrew/Cellar/rivet/3.1.10/share/Rivet/ATLAS_2011_I928289_W.yoda"
DATA_TAB = {"WP": "d05-x01-y04", "WM": "d03-x01-y04", "asym": "d07-x01-y04"}
# The ATLAS tables are d sigma / d eta averaged over the two signs of eta (their
# integral over 0-2.5 is HALF the fiducial cross section), while NNLOJET's
# abs_ylp histogram is d sigma / d|eta| = 2 x that.  Every comparison below is
# either unit-normalised or a ratio of the two charges, so the factor cancels;
# never compare the two absolutely without it.
SIGN = {"WP": "+", "WM": "-"}
FIG_SUFFIX = os.environ.get("W_FIG_SUFFIX", "")   # e.g. "_noctw" for the variant without the decay-angle tower

# decay-angle tower: W_CTW_MODE = "full" (Born tower over [-1,1], depth W_CTW_N <= 8),
# "window" (profiled tower over the bulk |cos th*| < 0.5-0.7, 6 orders + window rate,
# through add_profiled_recoil), or "off".  W_CTW=0 is a shorthand for "off".
CTW_MODE = os.environ.get("W_CTW_MODE", "off" if os.environ.get("W_CTW", "1") == "0" else "full")
USE_CTW = CTW_MODE == "full"
USE_CTW_WIN = CTW_MODE in ("window", "joint")
USE_CTW_JOINT = CTW_MODE == "joint"                    # + 16 joint <T_m(pT_W) T_n(cos th*)> moments
CTW_JOINT_N = int(os.environ.get("W_CTW_JOINT_N", "4"))
CTW_N = int(os.environ.get("W_CTW_N", "8"))           # depth of the full tower (moments booked to 8)
CTW_WIN_N = int(os.environ.get("W_CTW_WIN_N", "6"))   # depth of the windowed tower (booked to 6)
CTW_WIN = dict(a=0.3, b=0.4, c=1.6, d=1.7)            # mirrors eval_w_ctw (on x = cos th* + 1)
BORN_TAGS = {"m_w": "mw", "y_abs": "absyw"}
if USE_CTW:
    BORN_TAGS["ctw"] = "ctw"
LOADER_KW = dict(born_tags=BORN_TAGS, n_born={"m_w": 12, "y_abs": 12, "ctw": CTW_N}, n_recoil=20,
                 x_match=XM, x_hi=XMAP, soft_lo=SOFT, recoil_cfg_name="pt_w",
                 norm_born="norm_born", w0="prof_wptw_0", wtag="prof_wptw")


def cfg():
    born = {"m_w": {"range": (MLO, MHI), "map": ("bw", MW, GW)},
            "y_abs": {"range": (0.0, YHI), "map": "lin"}}
    if USE_CTW:
        born["ctw"] = {"range": (-1.0, 1.0), "map": "lin"}
    recoil = {"pt_w": {"range": (SOFT, XMAP), "map": "log", "soft_lo": SOFT,
                       "profile": {"a": XM, "b": XB, "c": XHI}}}
    if USE_CTW_WIN:
        # x = cos th* + 1 in (0, 2), linear map (= cos th* itself), window profile of eval_w_ctw
        recoil["ctw1"] = {"range": (0.0, 2.0), "map": "lin", "soft_lo": 0.0, "profile": dict(CTW_WIN)}
    mixed = {}
    if USE_CTW_JOINT:
        mixed["ptwctw"] = dict(observables=("pt_w", "ctw1"),
                               range={"pt_w": (SOFT, XMAP), "ctw1": (0.0, 2.0)},
                               map={"pt_w": "log", "ctw1": "lin"},
                               profile={"pt_w": dict(a=XM, b=XB, c=XHI, d=XHI), "ctw1": dict(CTW_WIN)},
                               n=CTW_JOINT_N)
    return dict(born=born, recoil=recoil, mixed=mixed, moment_selection=True)


def wdir(charge):
    return os.path.join(WROOT, charge)


def dens(x, w, e):
    m = np.isfinite(x)
    h, _ = np.histogram(x[m], e, weights=np.asarray(w, float)[m] / np.asarray(w, float).sum())
    return h / np.diff(e)


def load_prior(charge):
    """Fiducial events of one charge from the shower_w.py npz files, plus the
    LO fiducial cross section (pb) of the prior itself."""
    pat = os.path.join(PRIOR_DIR, "wp_*.npz" if charge == "WP" else "wm_*.npz")
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"no prior files for {charge}: {pat}")
    parts, n_tot, sw_fid = [], 0.0, 0.0
    for f in files:
        d = dict(np.load(f, allow_pickle=True))
        parts.append(d)
        n_tot += float(d["n_total"])
        sw_fid += float(d["sum_w_fid"])              # = N_fid * sigma_LHE  (unweighted LHE)
    sigma_fid_lo = sw_fid / n_tot                      # pb
    g = lambda k: np.concatenate([np.asarray(p[k], float) for p in parts])
    fid = g("in_fid") > 0.5
    l, nu = g("l_born"), g("nu")
    pw = l + nu
    with np.errstate(divide="ignore", invalid="ignore"):
        m2 = pw[:, 3] ** 2 - pw[:, 0] ** 2 - pw[:, 1] ** 2 - pw[:, 2] ** 2
        m = np.sqrt(np.clip(m2, 0, None)); qt = np.hypot(pw[:, 0], pw[:, 1])
        ctw = ((l[:, 3] + l[:, 2]) * (nu[:, 3] - nu[:, 2]) - (l[:, 3] - l[:, 2]) * (nu[:, 3] + nu[:, 2])) / (m * np.sqrt(m * m + qt * qt))
        ctw = np.clip(np.nan_to_num(np.where(pw[:, 2] < 0, -ctw, ctw)), -1, 1)   # CS z-axis along the W
    ev = dict(m_w=g("m_w"), y_abs=np.abs(g("y_w")), pt_w=g("pt_w"), eta_abs=np.abs(g("eta_l")),
              pt_l=g("pt_l"), mt=g("mt"), pt_nu=g("pt_nu"), ctw=ctw, ctw1=ctw + 1.0, weight=g("weight"))
    ev = {k: v[fid] for k, v in ev.items()}
    # keep only the expected lepton charge: a rare event whose hardest charged
    # lepton comes from a shower photon splitting carries the wrong sign
    want = +1 if charge == "WP" else -1
    m = np.isfinite(ev["m_w"]) & (ev["weight"] > 0) & (g("charge")[fid] == want)
    ev = {k: v[m] for k, v in ev.items()}
    nmax = int(os.environ.get("W_NMAX", 0))
    if nmax and len(ev["weight"]) > nmax:
        idx = np.random.default_rng(0).choice(len(ev["weight"]), nmax, replace=False)
        ev = {k: v[idx] for k, v in ev.items()}
    return ev, sigma_fid_lo, files


def band_weights(charge, ev, seeds, res):
    """6 scale re-solves + N_BOOT bootstrap re-solves, cached on disk, holding
    the central solve's moment counts fixed (fixed_orders)."""
    tag = ('_ctw%d' % CTW_N) if USE_CTW else (('_ctwjoint%d' % CTW_JOINT_N) if USE_CTW_JOINT else (('_ctwwin%d' % CTW_WIN_N) if USE_CTW_WIN else '_noctw'))
    cache = os.path.join(HERE, f"w_band_weights_{charge}{tag}.npz")
    if os.path.exists(cache):
        Z = dict(np.load(cache))
        if Z["central"].shape == res.weights.shape and np.allclose(Z["central"], res.weights, rtol=1e-6, atol=0):
            print(f"  band cache hit ({len(Z['scale_w'])} scale + {len(Z['boot_w'])} boot)")
            return Z["scale_w"], Z["boot_w"]
        print("  band cache stale; recomputing")
    lam0 = res.report["lam"]
    fixed = {o: int(n) for o, n in res.chosen_moments.items()}
    c = {**cfg(), "lam0": lam0, "fixed_orders": fixed}
    scale_w = []
    for s in range(1, 7):
        Ms = load_moments(charge, seeds, scale_idx=s)
        r_ = upgrade(ev, Ms, c)
        scale_w.append(r_.weights)
        print(f"    scale {s}: effN {100*r_.effN:.1f}%")
    rng = np.random.default_rng(1)
    boot_w = []
    for b in range(N_BOOT):
        sb = list(rng.choice(seeds, size=len(seeds), replace=True))
        Mb = load_moments(charge, sb)
        r_ = upgrade(ev, Mb, c)
        boot_w.append(r_.weights)
    scale_w, boot_w = np.array(scale_w), np.array(boot_w)
    np.savez_compressed(cache, central=res.weights, scale_w=scale_w, boot_w=boot_w)
    print(f"  cached -> {cache}")
    return scale_w, boot_w


def fo_abs(charge, channels, seeds, tag):
    """Absolute fixed-order histogram (channel-summed, seed-averaged): returns
    (lo, hi, values[nbin, nscale], seed_std[nbin]) in NNLOJET units (fb)."""
    acc = []
    for s in seeds:
        tot = None
        for ch in channels:
            r0 = _load(os.path.join(wdir(charge), f"ch_{ch}", f"{charge}.{RUN}.{ch}.{tag}.s{s}.dat"))
            if r0 is None:
                tot = None
                break
            lo, _, hi, v, _ = r0
            tot = v.copy() if tot is None else tot + v
        if tot is not None:
            acc.append(tot)
    acc = np.array(acc)
    return lo, hi, acc.mean(0), acc[:, :, 0].std(0, ddof=1) / np.sqrt(len(acc)) if len(acc) > 1 else 0 * acc[0, :, 0]


def sigma_fid_fo(charge, seeds, scale_idx=0):
    """NNLO fiducial cross section (fb) from the norm_born accumulator (a
    single-bin histogram: value x width)."""
    lo, hi, v, _ = fo_abs(charge, CH6, seeds, "norm_born")
    return float(v[0, scale_idx] * (hi[0] - lo[0]))


def native_edges(charge, tag):
    f = sorted(glob.glob(os.path.join(wdir(charge), "**", f"*.{tag}.s*.dat"), recursive=True))[0]
    rows = [l.split() for l in open(f) if l.strip() and not l.startswith("#")]
    lo = np.array([float(r[0]) for r in rows]); hi = np.array([float(r[2]) for r in rows])
    return np.concatenate([lo[:1], hi])


def panel(name, key, e, xlab, title, fo_tag, fo_ch, charge, ev, res, scale_w, boot_w,
          logx=False, anchor_window=None, ratio_ylim=(0.6, 1.4), anchor_mode="sharp",
          data=None, fo_label=None, legend_loc="lower left", logy=None):
    """One observable per figure (the ggh_eventlevel.py panel): main + ratio to
    fixed order, BAND = seven-point scale envelope, ERROR BARS = statistics.
    `data` = dict(lo, hi, y, ey) in absolute units overlays the measurement,
    unit-normalised over its own bins like every other curve."""
    bw = np.diff(e); ctr = np.sqrt(e[:-1] * e[1:]) if logx else 0.5 * (e[:-1] + e[1:])
    x = ev[key]
    sel = np.ones(len(x), bool)

    def nd(w):
        h = dens(x, w, e)
        s_ = (h * bw).sum()
        return h / s_ if s_ > 0 else h
    hp, hq = nd(ev["weight"]), nd(res.weights)
    hs = np.array([nd(w) for w in scale_w])
    q_lo, q_hi = np.minimum(hq, hs.min(0)), np.maximum(hq, hs.max(0))
    boot = np.array([nd(w) for w in boot_w]).std(0, ddof=1)
    fin = np.isfinite(x)
    qq = res.weights[fin] / res.weights.sum()
    s2, _ = np.histogram(x[fin], e, weights=qq ** 2)
    nrm_q = (dens(x, res.weights, e) * bw).sum()
    mc = np.sqrt(s2) / bw / max(nrm_q, 1e-30)
    q_err = np.hypot(boot, mc)

    seeds = common_seeds(wdir(charge), RUN, list(fo_ch), tag=fo_tag, prefix=charge)
    fb = fo_curve_band(wdir(charge), RUN, list(fo_ch), seeds, fo_tag, edges=e, prefix=charge, members=True)
    _, _, fd, _, _, fst, allc = fb
    ok = np.isfinite(fd) & (fd > 0)
    if anchor_window is not None:
        wlo, whi = anchor_window
        wok = ok & (ctr >= wlo) & (ctr < whi)
        jj = np.where(wok)[0]
        xlo_, xhi_ = (e[jj[0]], e[jj[-1] + 1]) if len(jj) else (wlo, whi)
        inw = (x >= xlo_) & (x < xhi_) & fin
        grid = (x >= e[0]) & (x < e[-1]) & fin
        if anchor_mode == "profile":
            wx = profile_w(x, XM, XB, XHI, XHI); wc = profile_w(ctr, XM, XB, XHI, XHI)
            tgt = float((res.weights * np.where(fin, wx, 0.0)).sum() / res.weights[grid].sum())
            k_ = tgt / np.nansum(np.where(ok, fd * wc, np.nan) * bw)
        else:
            tgt = float(res.weights[inw].sum() / res.weights[grid].sum())
            k_ = tgt / np.nansum(np.where(wok, fd, np.nan) * bw)
        fo = np.where(ok, fd, np.nan) * k_
        mem = np.array([np.where(ok, c_, np.nan) * k_ for c_ in allc])
        fo_st = fst * k_
    else:
        nrm = np.nansum(np.where(ok, fd, np.nan) * bw)
        fo = np.where(ok, fd, np.nan) / nrm
        mem = np.array([np.where(ok, c_, np.nan) for c_ in allc])
        mem = mem / np.nansum(mem * bw, axis=1, keepdims=True)
        fo_st = fst / nrm
    fo_lo = np.minimum(fo, np.nanmin(mem, 0)); fo_hi = np.maximum(fo, np.nanmax(mem, 0))

    seam_panel = anchor_window is not None
    if seam_panel:
        fig, ax = plt.subplots(3, 1, figsize=(6.8, 8.5),
                               gridspec_kw={"height_ratios": [2.1, 1.15, 0.78], "hspace": 0.07})
        a, r, u = ax
    else:
        fig, ax = plt.subplots(2, 1, figsize=(6.8, 7.6),
                               gridspec_kw={"height_ratios": [2.1, 1.15], "hspace": 0.07})
        a, r = ax; u = None
    order = "NNLO" if "RR" in CH6 else ("NLO" if "R" in CH6 else "LO")
    a.stairs(fo, e, color=C["fo"], ls=LS["fo"], lw=LW["fo"],
             label=fo_label or (rf"fixed order ({order})" if tuple(fo_ch) == tuple(CH6) else r"fixed order"))
    a.stairs(hp, e, color=C["prior"], ls="--", lw=2.0, label=r"PS+LO prior")
    a.stairs(hq, e, color=C["maxent"], lw=3.0, label=r"MaxEnt ($0\%\ w<0$)")
    ref = fo
    vok = ok.copy()
    half = 0.5 * (fo_hi - fo_lo) / np.where(ref > 0, ref, np.inf)
    fo_stat = fo_st / np.where(ref > 0, ref, np.inf)
    r.fill_between(ctr, np.where(vok, 1 - half, np.nan), np.where(vok, 1 + half, np.nan),
                   color=C["band"], alpha=0.55, step="mid", lw=0, label=r"FO scale")
    r.errorbar(stagger(e, 0, 3), np.where(vok, np.ones_like(ctr), np.nan),
               yerr=np.where(vok, fo_stat, np.nan), fmt="none", ecolor="0.35", elinewidth=1.0, capsize=1.6)
    r.fill_between(ctr, np.where(vok, q_lo / ref, np.nan), np.where(vok, q_hi / ref, np.nan),
                   color=C["maxent"], alpha=0.20, step="mid", lw=0)
    for h_, ckey, lw_ in ((hp, "prior", 1.8), (hq, "maxent", 2.4)):
        r.stairs(np.where(vok, h_ / ref, np.nan), e, color=C[ckey], ls=LS[ckey], lw=lw_)
    wp_ = ev["weight"][fin] / ev["weight"].sum()
    s2p, _ = np.histogram(x[fin], e, weights=wp_ ** 2)
    nrm_p = (dens(x, ev["weight"], e) * bw).sum()
    err_p = np.sqrt(s2p) / bw / max(nrm_p, 1e-30)
    r.errorbar(stagger(e, 1, 3), np.where(vok, hp / ref, np.nan), yerr=np.where(vok, err_p / ref, np.nan),
               fmt="none", ecolor=C["prior"], elinewidth=1.1, capsize=1.8)
    r.errorbar(stagger(e, 2, 3), np.where(vok, hq / ref, np.nan), yerr=np.where(vok, q_err / ref, np.nan),
               fmt="none", ecolor=C["maxent"], elinewidth=1.3, capsize=2.2)
    r.axhline(1, color="k", lw=0.8)
    med_data = None
    if data is not None:
        # the measurement, unit-normalised over its own bins (a shape comparison,
        # like every other curve here); drawn in both panels
        dl, dh, dy, de = data["lo"], data["hi"], data["y"], data["eyl"]
        dn = (dy * (dh - dl)).sum()
        yd, ed = dy / dn, de / dn
        xc = 0.5 * (dl + dh)
        a.errorbar(xc, yd, yerr=ed, xerr=0.5 * (dh - dl), fmt="o", color="k", ms=4.5, lw=1.2,
                   capsize=0, label=r"ATLAS", zorder=8)
        # ratio to FO on the data bins: FO rebinned onto them
        fbd = fo_curve_band(wdir(charge), RUN, list(fo_ch), seeds, fo_tag,
                            edges=np.concatenate([dl[:1], dh]), prefix=charge, members=False)
        fdd = fbd[2]; fdd = fdd / np.nansum(fdd * (dh - dl))
        r.errorbar(xc, yd / fdd, yerr=ed / fdd, fmt="o", color="k", ms=4.5, lw=1.2, capsize=0, zorder=8)
        # and the sample on the data bins, for the printed medians
        ed_ = np.concatenate([dl[:1], dh])
        hpd = dens(x, ev["weight"], ed_); hpd /= (hpd * (dh - dl)).sum()
        hqd = dens(x, res.weights, ed_); hqd /= (hqd * (dh - dl)).sum()
        med_data = (100 * np.median(np.abs(hpd / yd - 1)), 100 * np.median(np.abs(hqd / yd - 1)),
                    100 * np.median(np.abs(fdd / yd - 1)))
    if u is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rq = np.where(hp > 0, hq / hp, np.nan)
        u.stairs(rq, e, color=C["maxent"], lw=2.4)
        u.axhline(1, color="k", lw=0.8)
        below = np.isfinite(rq) & (ctr < XM)
        if below.sum() > 1:
            rb = rq[below]
            print(f"  {name}: below-seam MaxEnt/prior flat to {100*(rb.max()-rb.min())/rb.mean():.1f}%")
        u.set_ylabel(r"MaxEnt / prior", fontsize=13)
        u.set_xlabel(xlab)
        r.tick_params(labelbottom=False); r.set_xlabel("")
    if logx:
        a.set_xscale("log"); r.set_xscale("log")
        if u is not None:
            u.set_xscale("log")
    if logy if logy is not None else logx:
        a.set_yscale("log")
    if anchor_window is not None:
        rw = r.twinx()
        xs = np.geomspace(max(e[0], 1e-3), e[-1], 400)
        rw.plot(xs, profile_w(xs, XM, XB, XHI, XHI), color=C["seam"], lw=1.4, ls="-", alpha=0.75)
        rw.set_ylim(-0.03, 1.55); rw.set_yticks([0.0, 0.5, 1.0])
        rw.set_ylabel(r"profile $w$", fontsize=11, color=C["seam"], labelpad=1)
        rw.tick_params(axis="y", labelsize=9, colors=C["seam"], length=2.5)
        if logx:
            rw.set_xscale("log")
        rw.set_xlim(e[0], e[-1])
        for p_ in [p for p in (a, r, u) if p is not None]:
            p_.axvline(XM, color=C["seam"], lw=2.0, ls="--")
    a.tick_params(labelbottom=False)
    a.set_title(title)
    a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}X$")
    r.set_ylabel(r"ratio to fixed order")
    if u is None:
        r.set_xlabel(xlab)
    else:
        r.set_xlabel(""); r.tick_params(labelbottom=False)
    r.set_ylim(*ratio_ylim)
    a.set_xlim(e[0], e[-1]); r.set_xlim(e[0], e[-1])
    if u is not None:
        u.set_xlim(e[0], e[-1])
        fin_ = np.isfinite(rq)
        if fin_.any():
            span = max(abs(np.nanmax(rq[fin_]) - 1.0), abs(1.0 - np.nanmin(rq[fin_])))
            u.set_ylim(1 - 1.25 * span, 1 + 1.25 * span)
    a.legend(loc=legend_loc, fontsize=12)
    out = os.path.join(HERE, f"{name}{FIG_SUFFIX}.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
    m_ = vok & np.isfinite(hp) & (ref > 0)
    line = (f"  {name}: prior {100*np.median(np.abs(hp[m_]/ref[m_]-1)):5.1f}%   "
            f"MaxEnt {100*np.median(np.abs(hq[m_]/ref[m_]-1)):5.1f}%  vs FO")
    if med_data is not None:
        line += f"   | vs ATLAS: prior {med_data[0]:.1f}%  MaxEnt {med_data[1]:.1f}%  FO {med_data[2]:.1f}%"
    print(line + f"   -> {out}")


def asymmetry_figure(R):
    """Charge asymmetry A(|eta_l|) = (W+ - W-)/(W+ + W-) on the ATLAS bins:
    prior (LO rates and shapes), MaxEnt (NNLO fiducial rates from the input,
    MaxEnt shapes), fixed order (NNLO, direct), and the ATLAS measurement."""
    d = yoda_read(YODA, DATA_TAB["asym"])
    e = np.concatenate([d["lo"][:1], d["hi"]]); bw = np.diff(e); xc = 0.5 * (e[:-1] + e[1:])
    ab = {}
    for ch in ("WP", "WM"):
        ev, res, sig_lo, sig_fo, scale_w, boot_w = (R[ch][k] for k in ("ev", "res", "sig_lo", "sig_fo", "scale_w", "boot_w"))
        hp = dens(ev["eta_abs"], ev["weight"], e); hp = hp / (hp * bw).sum() * sig_lo        # pb / unit eta
        hq = dens(ev["eta_abs"], res.weights, e); hq = hq / (hq * bw).sum() * sig_fo[0] / 1e3
        hqs = np.array([dens(ev["eta_abs"], w, e) for w in scale_w])
        hqs = hqs / (hqs * bw).sum(1, keepdims=True) * (sig_fo[1:7] / 1e3)[:, None]
        hqb = np.array([dens(ev["eta_abs"], w, e) for w in boot_w])
        hqb = hqb / (hqb * bw).sum(1, keepdims=True) * sig_fo[0] / 1e3
        lo, hi, v, st = fo_abs(ch, CH6, R[ch]["seeds"], "etal_atlas")
        ab[ch] = dict(p=hp, q=hq, qs=hqs, qb=hqb, fo=v[:, 0] / 1e3, fos=v[:, 1:7].T / 1e3, fost=st / 1e3)
    A = lambda p, m: (p - m) / (p + m)
    ap = A(ab["WP"]["p"], ab["WM"]["p"]); aq = A(ab["WP"]["q"], ab["WM"]["q"])
    afo = A(ab["WP"]["fo"], ab["WM"]["fo"])
    aqs = np.array([A(ab["WP"]["qs"][i], ab["WM"]["qs"][i]) for i in range(6)])
    aqb = np.array([A(ab["WP"]["qb"][i], ab["WM"]["qb"][i]) for i in range(len(ab["WP"]["qb"]))]).std(0, ddof=1)
    afos = np.array([A(ab["WP"]["fos"][i], ab["WM"]["fos"][i]) for i in range(6)])
    # statistical error of the fixed-order asymmetry (seed spread, propagated)
    P_, M_ = ab["WP"]["fo"], ab["WM"]["fo"]; sP, sM = ab["WP"]["fost"], ab["WM"]["fost"]
    afost = np.sqrt((2 * M_ / (P_ + M_) ** 2) ** 2 * sP ** 2 + (2 * P_ / (P_ + M_) ** 2) ** 2 * sM ** 2)
    fig, (a, r) = plt.subplots(2, 1, figsize=(6.8, 7.6), gridspec_kw={"height_ratios": [2.1, 1.15], "hspace": 0.07})
    a.fill_between(xc, np.minimum(afo, afos.min(0)), np.maximum(afo, afos.max(0)), color=C["band"], alpha=0.55, step="mid", lw=0)
    a.stairs(afo, e, color=C["fo"], ls=LS["fo"], lw=LW["fo"], label=rf"fixed order ({'NNLO' if 'RR' in CH6 else 'NLO'})")
    a.stairs(ap, e, color=C["prior"], ls="--", lw=2.0, label=r"PS+LO prior")
    a.fill_between(xc, np.minimum(aq, aqs.min(0)), np.maximum(aq, aqs.max(0)), color=C["maxent"], alpha=0.20, step="mid", lw=0)
    a.stairs(aq, e, color=C["maxent"], lw=3.0, label=r"MaxEnt ($0\%\ w<0$)")
    a.errorbar(xc, aq, yerr=aqb, fmt="none", ecolor=C["maxent"], elinewidth=1.3, capsize=2.2)
    a.errorbar(stagger(e, 0, 3), afo, yerr=afost, fmt="none", ecolor=C["fo"], elinewidth=1.0, capsize=1.6)
    a.errorbar(d["x"], d["y"], yerr=d["eyl"], xerr=d["exl"], fmt="o", color="k", ms=4.5, lw=1.2, capsize=0, label=r"ATLAS", zorder=8)
    for h_, ckey, lw_ in ((ap, "prior", 1.8), (aq, "maxent", 2.4), (afo, "fo", 2.0)):
        r.stairs(h_ - d["y"], e, color=C[ckey], ls=LS[ckey], lw=lw_)
    r.fill_between(xc, aqs.min(0) - d["y"], aqs.max(0) - d["y"], color=C["maxent"], alpha=0.20, step="mid", lw=0)
    r.fill_between(xc, afos.min(0) - d["y"], afos.max(0) - d["y"], color=C["band"], alpha=0.55, step="mid", lw=0)
    r.errorbar(d["x"], 0 * d["y"], yerr=d["eyl"], fmt="o", color="k", ms=4.5, lw=1.2, capsize=0, zorder=8)
    r.errorbar(stagger(e, 0, 3), afo - d["y"], yerr=afost, fmt="none", ecolor=C["fo"], elinewidth=1.0, capsize=1.6)
    r.axhline(0, color="k", lw=0.8)
    a.set_title(r"$W$ charge asymmetry, predicted")
    a.set_ylabel(r"$A_\ell(|\eta_\ell|)$"); r.set_ylabel(r"difference to data"); r.set_xlabel(r"$|\eta_\ell|$")
    a.tick_params(labelbottom=False); a.set_xlim(e[0], e[-1]); r.set_xlim(e[0], e[-1])
    a.legend(loc="upper left", fontsize=12)
    out = os.path.join(HERE, f"fig_w_asym{FIG_SUFFIX}.pdf"); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
    chi = lambda h, s_=0.0: np.sqrt(np.mean(((h - d["y"]) / np.hypot(d["eyl"], s_)) ** 2))
    print(f"  fig_w_asym: rms pull vs ATLAS  prior {chi(ap):.2f}  MaxEnt {chi(aq):.2f}  FO {chi(afo):.2f}   "
          f"mean |A-A_data|  prior {np.mean(np.abs(ap-d['y'])):.4f}  MaxEnt {np.mean(np.abs(aq-d['y'])):.4f}  FO {np.mean(np.abs(afo-d['y'])):.4f}  -> {out}")
    print(f"  fig_w_asym: rms pull with the curve's own stat error folded in  MaxEnt {chi(aq, aqb):.2f}  FO {chi(afo, afost):.2f}  "
          f"(FO stat err last two bins {afost[-2]:.3f} {afost[-1]:.3f});  last-bin pull vs data  prior {(ap[-1]-d['y'][-1])/d['eyl'][-1]:+.2f}  "
          f"MaxEnt {(aq[-1]-d['y'][-1])/np.hypot(d['eyl'][-1], aqb[-1]):+.2f}  FO {(afo[-1]-d['y'][-1])/np.hypot(d['eyl'][-1], afost[-1]):+.2f}")


def main():
    check_seam(XM, MW, label="W")
    R = {}
    for ch in CHARGES:
        print(f"\n=== {ch} ===")
        seeds = common_seeds(wdir(ch), RUN, CH6, prefix=ch)
        print(f"  FO seeds usable ({len(seeds)}): {seeds[:6]}...  channels {CH6}")
        if not seeds:
            sys.exit(f"no complete seeds for {ch} in {wdir(ch)}")
        M = load_moments(ch, seeds)
        print("  <T_n(m_W)>   :", " ".join(f"{v:+.4f}" for v in M["born"]["m_w"]["values"][:6]), "...")
        print("  <T_n(|y_W|)> :", " ".join(f"{v:+.4f}" for v in M["born"]["y_abs"]["values"][:6]), "...")
        if USE_CTW:
            print("  <T_n(cos th*)>:", " ".join(f"{v:+.4f}" for v in M["born"]["ctw"]["values"][:6]), "...")
        if USE_CTW_WIN:
            rw = M["recoil"]["ctw1"]
            print("  <T_n(cos th*)>_w:", " ".join(f"{v:+.4f}" for v in rw["window_values"]), f"  window rate {rw['rate']:.4f}")
        if USE_CTW_JOINT:
            mx = M["mixed"]["ptwctw"]
            print(f"  joint <T_m(pT_W) T_n(cos th*)>: {len(mx['values'])} moments, joint rate {mx['rate']:.4f}")
        rc = M["recoil"]["pt_w"]
        print("  <T_n(pT_W)>_w:", " ".join(f"{v:+.4f}" for v in rc["window_values"][:6]), f"...  w-rate R = {rc['rate']:.4f}")
        ev, sig_lo, files = load_prior(ch)
        sig_fo = np.array([sigma_fid_fo(ch, seeds, s) for s in range(7)])
        print(f"  prior: {len(ev['weight']):,} fiducial events from {len(files)} files; "
              f"sigma_fid LO(prior) {sig_lo:.1f} pb, NNLO(input) {sig_fo[0]/1e3:.1f} pb "
              f"(scale {sig_fo[1:].min()/1e3:.1f}-{sig_fo[1:].max()/1e3:.1f})")
        res = upgrade(ev, M, cfg())
        print(f"  effN {100*res.effN:.1f}%  closure {res.closure:.2e}  neg-wt {100*np.mean(res.weights<=0):.1f}%  "
              f"chosen {dict(res.chosen_moments)}")
        scale_w, boot_w = band_weights(ch, ev, seeds, res)
        R[ch] = dict(ev=ev, res=res, sig_lo=sig_lo, sig_fo=sig_fo, scale_w=scale_w, boot_w=boot_w, seeds=seeds)
        sg = SIGN[ch]
        # constrained
        panel(f"fig_w_ptw_{ch}", "pt_w", np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 50, 60, 70, 80, 90, 100,
                                                    125, 150, 175, 200, 250, 300, 350, 400, 500], float),
              rf"$p_T^{{W^{sg}}}$ [GeV]", rf"$p_T^{{W^{sg}}}$, constrained above the seam", "ptw_full", CHJ,
              ch, ev, res, scale_w, boot_w, logx=True, anchor_window=(XM, 500.0), ratio_ylim=(0.6, 1.5),
              anchor_mode="profile", fo_label=(r"fixed order (NLO $W$+jet)" if "RR" in CH6 else r"fixed order (LO $W$+jet)"))
        panel(f"fig_w_yw_{ch}", "y_abs", np.linspace(0, 4, 21), rf"$|y_{{W^{sg}}}|$",
              rf"$|y_{{W^{sg}}}|$, constrained at NNLO", "absyw_fine", tuple(CH6), ch, ev, res, scale_w, boot_w,
              ratio_ylim=(0.8, 1.2))
        panel(f"fig_w_mw_{ch}", "m_w", np.linspace(40, 200, 41), rf"$m_{{W^{sg}}}$ [GeV]",
              rf"$m_{{W^{sg}}}$, constrained at NNLO", "mw_fine", tuple(CH6), ch, ev, res, scale_w, boot_w,
              ratio_ylim=(0.6, 1.4), logy=True, legend_loc="upper right")
        if USE_CTW or USE_CTW_WIN:
            panel(f"fig_w_ctw_{ch}", "ctw", np.linspace(-1, 1, 21), r"$\cos\theta^*$",
                  rf"$\cos\theta^*$ ($W^{sg}$), constrained at NNLO", "ctw_fine", tuple(CH6), ch, ev, res, scale_w, boot_w,
                  ratio_ylim=(0.8, 1.2), legend_loc="upper left")
        # followers
        d = yoda_read(YODA, DATA_TAB[ch])
        panel(f"fig_w_etal_{ch}", "eta_abs", np.concatenate([d["lo"][:1], d["hi"]]), r"$|\eta_\ell|$",
              rf"$|\eta_\ell|$ ($W^{sg}$), never constrained", "etal_atlas", tuple(CH6), ch, ev, res, scale_w, boot_w,
              ratio_ylim=(0.9, 1.1), data=d)
        panel(f"fig_w_ptl_{ch}", "pt_l", np.linspace(20, 120, 26), rf"$p_T^{{\ell}}$ [GeV]",
              rf"$p_T^{{\ell}}$ ($W^{sg}$), never constrained", "ptl_fine", tuple(CH6), ch, ev, res, scale_w, boot_w,
              ratio_ylim=(0.5, 1.5), logy=True, legend_loc="upper right")
        panel(f"fig_w_mt_{ch}", "mt", np.linspace(40, 200, 21), rf"$m_T$ [GeV]",
              rf"$m_T$ ($W^{sg}$), never constrained", "mt_fine", tuple(CH6), ch, ev, res, scale_w, boot_w,
              ratio_ylim=(0.5, 1.5), logy=True, legend_loc="upper right")
    if all(c in R for c in ("WP", "WM")):
        asymmetry_figure(R)


if __name__ == "__main__":
    main()
