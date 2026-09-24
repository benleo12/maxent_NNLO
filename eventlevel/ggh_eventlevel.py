#!/usr/bin/env python3
r"""gg -> H upgraded with EVENT-LEVEL NNLOJET moments, all six channels (NNLO).

Constrained : |y_H| (Born, lin [0,4]) and pT_H (recoil, log [1,1000] with the
              smooth profile [37,74]->1000, exactly as booked in NNLOJET).
Followers   : the anti-kT R=0.4 jet observables pT_j1 and |y_j1| carried by
              the v2 prior -- never constrained, predicted through the
              prior's correlations, validated against the ptj1_all and
              absyj1_w37 references of the ggh_moments2 production.
Uncertainty : the full band program -- MaxEnt scale envelope from 6
              warm-started re-solves + bootstrap-over-seeds bars; FO
              references carry their 7-point scale envelope (+) seed scatter.
              Variant weights cached in ggh_band_weights.npz.

This is the INCLUSIVE-mode demonstration: no cuts anywhere, one upgraded
sample, any later selection at the prior's acceptance accuracy.

Figures: fig_ggh_pT_H, fig_ggh_y_abs, fig_ggh_ptj1, fig_ggh_yj1.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import use_pub_style, C, LS, LW, rebin_density
use_pub_style(base=17)
from maxent_upgrade import upgrade, check_seam, profile_w
from nnlojet_moments import (fo_moments_smooth_from_nnlojet, common_seeds,
                             fo_curve_band)
from bandviz import stagger

GDIR = os.path.join(os.environ.get("NNLOJET_ROOT",
        os.path.expanduser("~/nnlojet-v1.0.2")), "ggh_moments2")
RUN, PREFIX = "GGH_MOMENTS", "H"
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
CHJ = ("R", "RR", "RV")               # channels with partons -> jets, pT_H>0
# Mirrors the profile COMPILED INTO NNLOJET (eval_w_pth: pa=37, pb=74,
# pc=pd=1000).  37 GeV is the power-counting seam for Q = m_H (check_seam).
XM, XB, XHI, SOFT = 37.0, 74.0, 1000.0, 1.0
Q_HARD = 125.0
YHI = 4.0
N_BOOT = 20

LOADER_KW = dict(born_tags={"y_abs": "yh"}, n_born=6, n_recoil=12,
                 x_match=XM, x_hi=XHI, soft_lo=SOFT,
                 recoil_cfg_name="pT_H", norm_born="norm_born",
                 w0="prof_wpt_0", wtag="prof_wpt", prefix=PREFIX)


def cfg():
    return dict(born={"y_abs": {"range": (0.0, YHI), "map": "lin"}},
                recoil={"pT_H": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                                 "profile": {"a": XM, "b": XB, "c": XHI}}},
                moment_selection=False)


def dens(x, w, e):
    m = np.isfinite(x)
    h, _ = np.histogram(x[m], e, weights=np.asarray(w, float)[m] / np.asarray(w, float).sum())
    return h / np.diff(e)


def load_prior():
    parts = [dict(np.load(os.path.join(HERE, f), allow_pickle=True))
             for f in ("ggh_prior_v2_s1.npz", "ggh_prior_v2_s2.npz")
             if os.path.exists(os.path.join(HERE, f))]
    if not parts:
        sys.exit("no v2 ggH prior found (need ggh_prior_v2_s*.npz)")
    g = lambda k: np.concatenate([np.asarray(p[k], float) for p in parts])
    ev = dict(pT_H=g("pT_H"), y_abs=np.abs(g("y_H")), weight=g("weight"),
              pT_j1=g("pT_j1"), y_j1=np.abs(g("y_j1")))
    m = np.isfinite(ev["pT_H"]) & np.isfinite(ev["weight"]) & (ev["weight"] > 0)
    ev = {k: v[m] for k, v in ev.items()}
    n = len(ev["weight"])
    idx = np.random.default_rng(0).choice(n, min(1_200_000, n), replace=False)
    return {k: v[idx] for k, v in ev.items()}, idx


def band_weights(ev, seeds, res):
    """6 scale re-solves + N_BOOT bootstrap re-solves, cached on disk."""
    cache = os.path.join(HERE, "ggh_band_weights.npz")
    if os.path.exists(cache):
        Z = dict(np.load(cache))
        if np.allclose(Z["central"], res.weights, rtol=1e-6, atol=0):
            print(f"  band cache hit ({len(Z['scale_w'])} scale + {len(Z['boot_w'])} boot)")
            return Z["scale_w"], Z["boot_w"]
        print("  band cache stale; recomputing")
    lam0 = res.report["lam"]
    scale_w = []
    for s in range(1, 7):
        Ms = fo_moments_smooth_from_nnlojet(GDIR, RUN, CH6, seeds,
                                            scale_idx=s, **LOADER_KW)
        r_ = upgrade(ev, Ms, {**cfg(), "lam0": lam0})
        scale_w.append(r_.weights)
        print(f"    scale {s}: effN {100*r_.effN:.1f}%")
    rng = np.random.default_rng(1)
    boot_w = []
    for b in range(N_BOOT):
        sb = list(rng.choice(seeds, size=len(seeds), replace=True))
        Mb = fo_moments_smooth_from_nnlojet(GDIR, RUN, CH6, sb, **LOADER_KW)
        r_ = upgrade(ev, Mb, {**cfg(), "lam0": lam0})
        boot_w.append(r_.weights)
    scale_w, boot_w = np.array(scale_w), np.array(boot_w)
    np.savez_compressed(cache, central=res.weights, scale_w=scale_w, boot_w=boot_w)
    print(f"  cached -> {cache}")
    return scale_w, boot_w


def panel(name, key, e, xlab, title, fo_tag, fo_ch, ev, res, scale_w, boot_w,
          logx=False, anchor_window=None, sample_cut=None, ratio_ylim=(0.6, 1.4), anchor_mode="sharp"):
    r"""One observable per figure: main + ratio-to-FO, full band treatment.

    anchor_window=(lo,hi): FO (and all its scale members) anchored to the
    MaxEnt rate in that window -- for spectra whose low end FO cannot
    describe.  Otherwise both sides are unit-normalized (shape comparison,
    scale members self-normalized BEFORE the envelope).
    sample_cut: mask applied to the sample arrays (e.g. pT_j1>=37 to mirror
    the FO histogram's own selector).
    """
    bw = np.diff(e); ctr = np.sqrt(e[:-1] * e[1:]) if logx else 0.5 * (e[:-1] + e[1:])
    x = ev[key]
    sel = np.ones(len(x), bool) if sample_cut is None else sample_cut
    def nd(w):
        h = dens(np.where(sel, x, np.nan), w, e)
        s_ = (h * bw).sum()
        return h / s_ if s_ > 0 else h
    hp, hq = nd(ev["weight"]), nd(res.weights)
    hs = np.array([nd(w) for w in scale_w])
    q_lo, q_hi = np.minimum(hq, hs.min(0)), np.maximum(hq, hs.max(0))
    boot = np.array([nd(w) for w in boot_w]).std(0, ddof=1)
    qq = res.weights[sel & np.isfinite(x)]; qq = qq / res.weights.sum()
    s2, _ = np.histogram(x[sel & np.isfinite(x)], e, weights=qq ** 2)
    nrm_q = (dens(np.where(sel, x, np.nan), res.weights, e) * bw).sum()
    mc = np.sqrt(s2) / bw / max(nrm_q, 1e-30)
    q_err = np.hypot(boot, mc)
    # ONE convention everywhere: BAND = scale envelope, CANDLE = statistics.
    # q_lo/q_hi stay the scale envelope; q_err is drawn as error bars below.

    fb = fo_curve_band(GDIR, RUN, fo_ch,
                       common_seeds(GDIR, RUN, fo_ch, tag=fo_tag, prefix=PREFIX),
                       fo_tag, edges=e, prefix=PREFIX, members=True)
    _, _, fd, _, _, fst, allc = fb
    ok = np.isfinite(fd) & (fd > 0)
    if anchor_window is not None:
        wlo, whi = anchor_window
        fin = sel & np.isfinite(x)
        # Anchor the fixed order to the sample's rate in the window.  Two
        # alignments have to hold or the whole FO curve is displaced:
        #  (i) same normalization basis -- each curve is unit-normalized over
        #      the plotted grid, so the target is the weight fraction WITHIN
        #      the grid (jet followers put ~40% of the sample, the no-jet
        #      events, outside the grid entirely);
        # (ii) same window -- the FO integral runs over whole bins, so the
        #      sample fraction must run over exactly those bins' edges.  Using
        #      the nominal x_match instead lets the sliver between the seam and
        #      the first window bin edge into the sample side only, which
        #      inflates the target and pushes the entire ratio below one.
        wok = ok & (ctr >= wlo) & (ctr < whi)
        jj = np.where(wok)[0]
        xlo_, xhi_ = (e[jj[0]], e[jj[-1] + 1]) if len(jj) else (wlo, whi)
        inw = (x >= xlo_) & (x < xhi_) & fin
        grid = (x >= e[0]) & (x < e[-1]) & fin
        if anchor_mode == "profile":
            # Anchor on the PROFILE-WEIGHTED window rate.  That rate is the
            # n=0 member of the recoil tower, so it is identical on the two
            # sides by construction: there is no freedom in this normalization.
            # The sharp-window anchor instead lets the ramp region, which holds
            # ~20% of the sample at 1.4-1.5x fixed order because the shower is
            # preserved there by design, force a compensating deficit where
            # w=1 -- an artifact of the comparison, not of the matching.
            wx = profile_w(x, XM, XB, XHI, XHI); wc = profile_w(ctr, XM, XB, XHI, XHI)
            tgt = float((res.weights * np.where(fin, wx, 0.0)).sum() / res.weights[grid].sum())
            k_ = tgt / np.nansum(np.where(ok, fd * wc, np.nan) * bw)
        else:
            tgt = float(res.weights[inw].sum() / res.weights[grid].sum())
            k_ = tgt / np.nansum(np.where(wok, fd, np.nan) * bw)
        fo = np.where(ok, fd, np.nan) * k_
        mem = np.array([np.where(ok, c, np.nan) * k_ for c in allc])
        fo_st = fst * k_
    else:
        nrm = np.nansum(np.where(ok, fd, np.nan) * bw)
        fo = np.where(ok, fd, np.nan) / nrm
        mem = np.array([np.where(ok, c, np.nan) for c in allc])
        mem = mem / np.nansum(mem * bw, axis=1, keepdims=True)
        fo_st = fst / nrm
    fo_lo = np.minimum(fo, np.nanmin(mem, 0)); fo_hi = np.maximum(fo, np.nanmax(mem, 0))

    seam_panel = anchor_window is not None
    if seam_panel:
        fig, ax = plt.subplots(3, 1, figsize=(6.8, 8.5),
                               gridspec_kw={"height_ratios": [2.1, 1.15, 0.78],
                                            "hspace": 0.07})
        a, r, u = ax
    else:
        fig, ax = plt.subplots(2, 1, figsize=(6.8, 7.6),
                               gridspec_kw={"height_ratios": [2.1, 1.15],
                                            "hspace": 0.07})
        a, r = ax
        u = None
    a.stairs(fo, e, color=C["fo"], ls=LS["fo"], lw=LW["fo"],
             label=r"fixed order (NLO $H$+jet)" if fo_ch == CHJ else r"fixed order (NNLO)")
    a.stairs(hp, e, color=C["prior"], ls="--", lw=2.0, label=r"PS+LO prior")
    a.stairs(hq, e, color=C["maxent"], lw=3.0, label=r"MaxEnt ($0\%\ w<0$)")
    ref = fo
    # Nothing is drawn faded.  This panel asks one question -- does the sample
    # agree with fixed order where fixed order is a valid reference -- and it
    # is drawn only there, at full strength.  Below the seam fixed order is not
    # a prediction at all (no resummation; the spectrum collapses as pT -> 0),
    # so a ratio to it would measure the reference's failure, not the sample's.
    # The region below the seam is not dropped from the figure: it gets the
    # bottom panel, where the sample is compared with the shower it must
    # reproduce there.
    # ratio to fixed order drawn EVERYWHERE fixed order is positive, below the
    # seam included: there it is not a prediction and simply leaves the panel
    vok = ok.copy()
    half = 0.5 * (fo_hi - fo_lo) / np.where(ref > 0, ref, np.inf)     # BAND
    fo_stat = fo_st / np.where(ref > 0, ref, np.inf)                   # CANDLE
    r.fill_between(ctr, np.where(vok, 1 - half, np.nan), np.where(vok, 1 + half, np.nan),
                   color=C["band"], alpha=0.55, step="mid", lw=0,
                   label=r"FO scale")
    r.errorbar(stagger(e, 0, 3), np.where(vok, np.ones_like(ctr), np.nan),
               yerr=np.where(vok, fo_stat, np.nan), fmt="none",
               ecolor="0.35", elinewidth=1.0, capsize=1.6)
    r.fill_between(ctr, np.where(vok, q_lo / ref, np.nan),
                   np.where(vok, q_hi / ref, np.nan),
                   color=C["maxent"], alpha=0.20, step="mid", lw=0)
    for h_, ckey, lw_ in ((hp, "prior", 1.8), (hq, "maxent", 2.4)):
        r.stairs(np.where(vok, h_ / ref, np.nan), e, color=C[ckey], ls=LS[ckey], lw=lw_)
    # only the prior needs bars: its uncertainty is purely its own statistics,
    # while the MaxEnt and fixed-order uncertainties are carried by their bands
    wp_ = ev["weight"][sel & np.isfinite(x)] / ev["weight"].sum()
    s2p, _ = np.histogram(x[sel & np.isfinite(x)], e, weights=wp_ ** 2)
    nrm_p = (dens(np.where(sel, x, np.nan), ev["weight"], e) * bw).sum()
    err_p = np.sqrt(s2p) / bw / max(nrm_p, 1e-30)
    r.errorbar(stagger(e, 1, 3), np.where(vok, hp / ref, np.nan),
               yerr=np.where(vok, err_p / ref, np.nan), fmt="none",
               ecolor=C["prior"], elinewidth=1.1, capsize=1.8)
    r.errorbar(stagger(e, 2, 3), np.where(vok, hq / ref, np.nan),
               yerr=np.where(vok, q_err / ref, np.nan), fmt="none",
               ecolor=C["maxent"], elinewidth=1.3, capsize=2.2)
    r.axhline(1, color="k", lw=0.8)

    # ---- the constraint's own strength, on the same axis -------------------
    # The recoil constraint does not switch on at the seam, it ramps: the
    # profile w is zero below x_m and reaches one only at x_b.  Plotting w
    # here turns the first window bin from an apparent defect into the
    # statement it actually is -- the sample still carries the shower where
    # w is small, and moves onto fixed order exactly as w -> 1.
    if anchor_window is not None:
        rw = r.twinx()
        xs = np.geomspace(max(e[0], 1e-3), e[-1], 400)
        rw.plot(xs, profile_w(xs, XM, XB, XHI, XHI), color=C["seam"], lw=1.4,
                ls="-", alpha=0.75)
        rw.set_ylim(-0.03, 1.55)
        rw.set_yticks([0.0, 0.5, 1.0])
        rw.set_ylabel(r"profile $w$", fontsize=11, color=C["seam"], labelpad=1)
        rw.tick_params(axis="y", labelsize=9, colors=C["seam"], length=2.5)
        if logx:
            rw.set_xscale("log")
        rw.set_xlim(e[0], e[-1])

    # ---- what the upgrade did to the prior, across the whole spectrum ------
    # Below the seam the recoil profile is identically zero, so the recoil
    # constraint cannot act there and the shower is kept: any departure from
    # one is the Born constraint reweighting events through its correlations,
    # not the recoil constraint reaching below its window.
    if u is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rq = np.where(hp > 0, hq / hp, np.nan)
        u.stairs(rq, e, color=C["maxent"], lw=2.4)
        u.axhline(1, color="k", lw=0.8)
        # a FLAT ratio below the seam is the statement: the shower's shape is
        # carried through untouched there (the level differs only because both
        # curves are normalized over the whole spectrum, and the upgrade moves
        # rate into the tail).  Quote the flatness so the reader can check it.
        below = np.isfinite(rq) & (ctr < XM)
        if below.sum() > 1:
            rb = rq[below]
            print(f"  {name}: below-seam MaxEnt/prior flat to {100*(rb.max()-rb.min())/rb.mean():.1f}%")
        u.set_ylabel(r"MaxEnt / prior", fontsize=13)
        u.set_xlabel(xlab)
        r.tick_params(labelbottom=False); r.set_xlabel("")
    if logx:
        a.set_xscale("log"); r.set_xscale("log"); a.set_yscale("log")
        if u is not None:
            u.set_xscale("log")
    if anchor_window is not None:
        # The constraint does not switch on at the seam, it RAMPS: the profile
        # is zero below x_m, rises smoothly across [x_m, x_b] and is one above.
        # Shading the whole window uniformly implies a constraint that is fully
        # active at 37 GeV, which it is not -- the ramp is why the first window
        # bins still track the prior.
        for p_ in [p for p in (a, r, u) if p is not None]:
            p_.axvline(XM, color=C["seam"], lw=2.0, ls="--")
    a.tick_params(labelbottom=False)
    a.set_title(title)
    a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}X$")
    r.set_ylabel(r"ratio to fixed order")
    if u is None:                      # the bottom panel carries the x-axis
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
    a.legend(loc="lower left", fontsize=12)
    out = os.path.join(HERE, f"{name}.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
    m_ = vok & np.isfinite(hp) & (ref > 0)
    print(f"  {name}: prior {100*np.median(np.abs(hp[m_]/ref[m_]-1)):5.1f}%   "
          f"MaxEnt {100*np.median(np.abs(hq[m_]/ref[m_]-1)):5.1f}%   -> {out}")


def native_edges(tag):
    """The fixed-order histogram's own bin edges.  Every ratio panel whose
    denominator is fixed order MUST use these: overlap-rebinning a coarser
    histogram onto finer plot bins produces a staircase, and a smooth curve
    divided by a staircase alternates bin by bin."""
    import glob as _g
    f = sorted(_g.glob(GDIR + f"/**/*.{tag}.s*.dat", recursive=True))[0]
    rows = [l.split() for l in open(f) if l.strip() and not l.startswith("#")]
    lo = np.array([float(r[0]) for r in rows]); hi = np.array([float(r[2]) for r in rows])
    return np.concatenate([lo[:1], hi])

def main():
    check_seam(XM, Q_HARD, label="gg->H")
    seeds = common_seeds(GDIR, RUN, CH6, prefix=PREFIX)
    print(f"gg->H FO seeds usable ({len(seeds)}): {seeds[:5]}...")
    M = fo_moments_smooth_from_nnlojet(GDIR, RUN, CH6, seeds, **LOADER_KW)
    print("  <T_n(|y_H|)> :", " ".join(f"{v:+.4f}" for v in M["born"]["y_abs"]["values"]))
    rc = M["recoil"]["pT_H"]
    print("  <T_n(pT_H)>_w:", " ".join(f"{v:+.4f}" for v in rc["window_values"][:6]), "...")
    print(f"  w-rate R = {rc['rate']:.4f}")

    ev, _ = load_prior()
    print(f"  prior events: {len(ev['weight']):,}  "
          f"(jet fraction {100*np.mean(ev['pT_j1'] > 20):.1f}%)")
    print("solving (all six channels: NNLO input) ...", flush=True)
    res = upgrade(ev, M, cfg())
    print(f"  effN {100*res.effN:.1f}%  closure {res.closure:.2e}  "
          f"neg-wt {100*np.mean(res.weights<=0):.1f}%")
    scale_w, boot_w = band_weights(ev, seeds, res)

    panel("fig_ggh_pT_H", "pT_H", np.geomspace(1, 500, 30), r"$p_T^{H}$ [GeV]",
          r"$p_T^{H}$, constrained above the seam", "pth_all", CHJ,
          ev, res, scale_w, boot_w, logx=True, anchor_window=(XM, 500.0),
          ratio_ylim=(0.6, 1.5), anchor_mode="profile")
    panel("fig_ggh_y_abs", "y_abs", native_edges("absyh_a"), r"$|y_H|$",
          r"$|y_H|$, constrained at NNLO", "absyh_a", tuple(CH6),
          ev, res, scale_w, boot_w, ratio_ylim=(0.8, 1.2))
    # ---- jet followers: never constrained, predicted through the prior ----
    panel("fig_ggh_ptj1", "pT_j1", np.geomspace(20, 500, 25), r"$p_T^{j_1}$ [GeV]",
          r"$p_T^{j_1}$, never constrained", "ptj1_all", CHJ,
          ev, res, scale_w, boot_w, logx=True, anchor_window=(XB, 500.0),
          ratio_ylim=(0.6, 1.5))
    panel("fig_ggh_yj1", "y_j1", np.linspace(0, 4.4, 23), r"$|y_{j_1}|$",
          r"$|y_{j_1}|\ (p_T^{j_1}>37$ GeV$)$, never constrained", "absyj1_w37", CHJ,
          ev, res, scale_w, boot_w, sample_cut=(ev["pT_j1"] >= 37.0),
          ratio_ylim=(0.6, 1.4))


if __name__ == "__main__":
    main()
