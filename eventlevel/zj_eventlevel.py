#!/usr/bin/env python3
r"""Drell-Yan + jet upgraded with EVENT-LEVEL NNLOJET moments.

NNLOJET calls this process ZJ and the code follows that, but it is the
LEPTONIC final state throughout -- decay_type = 1, and every observable
here is the dilepton system (mll, |y_ll|, pi - dphi_ll) plus the jets.
It is the same process the papers call Drell-Yan + jet; the figures say so,
so that one process is not named two ways across the two papers.

Constrained : m_ll, |y_ll| (Born) and pT_j1 (recoil, smooth profile [30,60]->1000)
Predicted   : pT_j2 and Delta phi(l1,l2)  -- both have their own event-level FO
              moments booked, so we can check the prediction against fixed order
              WITHOUT having fitted them.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import (use_pub_style, C, LS, LW, FADE, support_mask,
                      shade_unsupported, rebin_density)
use_pub_style(base=17)
from maxent_upgrade import upgrade, check_seam
from nnlojet_moments import (fo_moments_smooth_from_nnlojet, common_seeds,
                             _load, _moment_over_seeds, add_profiled_recoil,
                             add_mixed_moments, fo_curve_band, MIRRORED_TAGS)
from bandviz import stagger

ZDIR = "/Users/user/nnlojet-v1.0.2/zj_moments2"   # clipped-map binary, mll = BW map
RUN, PREFIX = "ZJ_MOMENTS", "ZJ"
# NLO (LO+R+V) by default.  The full NNLO set (LO,R,V,RRa,RRb,RV,VV) exists at
# 20 seeds and is selectable with ZJ_CH, but at the statistics reachable here its
# moments are not precise enough to constrain: <T_1(pT_j1)> comes out at
# SNR 2.8 against 45 at NLO, and <T_3> at 3.0 against 51.  Imposing moments at
# SNR ~3 injects more noise than information -- closure degrades tenfold
# (9.6e-4 -> 9.5e-3) and the pi-dphi prediction falls from 9% to 32%.  Reaching
# NLO-like precision at NNLO needs 1e2-1e3 times the statistics.
CH = os.environ.get("ZJ_CH", "LO,R,V").split(",")
NMIX = int(os.environ.get("NMIX", 4))      # mixed orders m,n = 1..NMIX
# mirrors eval_w_ptj1 (pa=30, pb=60)
XM, XB, XHI, SOFT = 30.0, 60.0, 1000.0, 10.0   # pT_j1 profile + log-map floor
N_DPHI, DPHI_A, DPHI_B = 4, 0.05, 0.20        # booked dphi tower: depth, compiled profile
C_DPHI = "#2b6ca3"                              # colour of the +dphi-tower state
Q_HARD = 91.1876


def cheb(u, n):
    u = np.clip(u, -1, 1)
    if n == 0: return np.ones_like(u)
    if n == 1: return u
    t0, t1 = np.ones_like(u), u
    for _ in range(2, n + 1): t0, t1 = t1, 2 * u * t1 - t0
    return t1


def prof_w(x, a, b, c):
    lx = np.log(np.maximum(x, 1e-30))
    t = np.clip((lx - np.log(a)) / (np.log(b) - np.log(a)), 0, 1)
    return t * t * t * (t * (t * 6 - 15) + 10) * (x < c)


def fo_moment(tag_base, w0, nmax, seeds):
    """FO <T_n> for a profiled observable (prediction check)."""
    out = []
    for n in range(1, nmax + 1):
        p, _ = _moment_over_seeds(ZDIR, RUN, CH, seeds, f"{tag_base}_{n}", w0, PREFIX)
        out.append(float(p[0]) if p is not None else np.nan)
    return np.array(out)


def main():
    check_seam(XM, Q_HARD, label="DY+jet")
    seeds = common_seeds(ZDIR, RUN, CH, prefix=PREFIX)
    print(f"DY+jet FO seeds usable: {seeds}")
    M = fo_moments_smooth_from_nnlojet(
        ZDIR, RUN, CH, seeds,
        born_tags={"mll": "mll", "y_abs": "absyz"},
        n_born=6, n_recoil=12, x_match=XM, x_hi=XHI, soft_lo=SOFT,
        recoil_cfg_name="ptj1", norm_born="norm_born",
        w0="prof_wj1_0", wtag="prof_wj1", prefix=PREFIX)
    rc = M["recoil"]["ptj1"]
    print("  <T_n(m_ll)>  :", " ".join(f"{v:+.4f}" for v in M['born']['mll']['values']))
    print("  <T_n(pT_j1)>_w:", " ".join(f"{v:+.4f}" for v in rc['window_values'][:6]), "...")
    print(f"  w-rate R = {rc['rate']:.4f}")

    z = np.load(os.path.join(HERE, "zj_prior.npz"))
    sel = (np.asarray(z["njet"]) >= 1) & np.isfinite(z["weight"]) & (z["weight"] > 0)
    ev = dict(mll=np.asarray(z["mll"])[sel], y_abs=np.asarray(z["y_abs"])[sel],
              ptj1=np.asarray(z["ptj1"])[sel], ptj2=np.asarray(z["ptj2"])[sel],
              pimdphi=np.asarray(z["pimdphi"])[sel], weight=np.asarray(z["weight"])[sel])
    print(f"  prior events (>=1 jet): {len(ev['weight']):,}")

    # Constrain BOTH jet spectra and, where the mixed moments exist, their JOINT
    # distribution.  pT_j1 and pT_j2 separately fix only the two marginals; it is
    # the correlation that determines pi - dphi_ll, which is left as the
    # prediction.  This is the case mixed moments were built for.
    recoil = {"ptj1": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                       "profile": {"a": XM, "b": XB, "c": XHI}}}
    mixed = {}
    if common_seeds(ZDIR, RUN, CH, tag="prof_wj12_0", prefix=PREFIX):
        sd2 = common_seeds(ZDIR, RUN, CH,
                           tag=["prof_wj12_0"] + [f"prof_wj12_{m}{n}"
                                                 for m in range(1, NMIX + 1)
                                                 for n in range(1, NMIX + 1)],
                           prefix=PREFIX)
        add_profiled_recoil(M, ZDIR, RUN, CH, sd2, "ptj2", wtag="prof_wj2",
                            w0="prof_wj2_0", n_recoil=6, x_match=XM, x_hi=XHI,
                            soft_lo=SOFT, prefix=PREFIX)
        add_mixed_moments(M, ZDIR, RUN, CH, sd2, "ptj12", wtag="prof_wj12",
                          w0="prof_wj12_0", n_max=NMIX, prefix=PREFIX)
        recoil["ptj2"] = {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                          "profile": {"a": XM, "b": XB, "c": XHI}}
        mixed["ptj12"] = dict(observables=("ptj1", "ptj2"),
                              range={"ptj1": (SOFT, XHI), "ptj2": (SOFT, XHI)},
                              map={"ptj1": "log", "ptj2": "log"},
                              profile={"ptj1": dict(a=XM, b=XB, c=XHI, d=XHI),
                                       "ptj2": dict(a=XM, b=XB, c=XHI, d=XHI)},
                              n=NMIX)
        nm = len(M["mixed"]["ptj12"]["values"])
        print(f"  mixed <T_m(pTj1) T_n(pTj2)>: {nm} moments, {len(sd2)} seeds, "
              f"R = {M['mixed']['ptj12']['rate']:.4f}")
    else:
        print("  mixed moments NOT present -- pT_j1 only, pT_j2 stays a prediction")
    cfg = dict(born={"mll": {"range": (66., 116.), "map": "bw"},
                     "y_abs": {"range": (0., 2.4), "map": "lin"}},
               recoil=recoil, mixed=mixed,
               followers=["pimdphi"] + ([] if mixed else ["ptj2"]),
               moment_selection=False)
    print("solving ...", flush=True)
    res = upgrade(ev, M, cfg)
    print(f"  effN {100*res.effN:.1f}%  closure {res.closure:.2e}  "
          f"neg-wt {100*np.mean(res.weights<=0):.1f}%")
    # THIRD STATE of the follower test: impose the Delta-phi tower NNLOJET
    # already books (prof_wdphi: profile in pi-dphi over [0.05,0.20], log map
    # on [0.01,pi]).  The marginal+cross tilt over-transports pi-dphi (0.87 ->
    # 1.49 against fixed order); this shows the failure is fixable by adding
    # the missing measurement, at ~1 point of effN.  Caveat carried into the
    # paper: that compiled window sits below the seam image ~0.67.
    res_dphi = None
    sdd = common_seeds(ZDIR, RUN, CH, tag=[f"prof_wdphi_{n}" for n in range(0, N_DPHI + 1)],
                       prefix=PREFIX)
    if sdd:
        import copy
        M2, cfg2 = copy.deepcopy(M), copy.deepcopy(cfg)
        add_profiled_recoil(M2, ZDIR, RUN, CH, sdd, "pimdphi", wtag="prof_wdphi",
                            w0="prof_wdphi_0", n_recoil=N_DPHI, x_match=DPHI_A, x_hi=np.pi,
                            soft_lo=0.01, prefix=PREFIX)
        cfg2["recoil"]["pimdphi"] = {"range": (0.01, np.pi), "map": "log", "soft_lo": 0.01,
                                     "profile": {"a": DPHI_A, "b": DPHI_B, "c": 4.0}}
        cfg2["followers"] = []
        res_dphi = upgrade(ev, M2, cfg2)
        print(f"  + Delta-phi tower (n<={N_DPHI}): effN {100*res_dphi.effN:.1f}%  "
              f"closure {res_dphi.closure:.2e}")

    # ---- prediction check: pT_j2 and (pi-dphi_ll) moments vs their FO values
    print("\nPREDICTED observables (never constrained) -- <T_n> prior / MaxEnt / FO")
    checks = [("ptj2", "prof_wj2", "prof_wj2_0", 6, (XM, XB, XHI)),
              ("pimdphi", "prof_wdphi", "prof_wdphi_0", 6, (0.05, 0.20, 4.0))]
    for key, base, w0, nmax, (a, b, c) in checks:
        fo = fo_moment(base, w0, nmax, seeds)
        x = ev[key]
        wv = prof_w(x, a, b, c)
        lo_m, hi_m = (SOFT, XHI) if key == "ptj2" else (0.01, np.pi)
        u = 2 * (np.log(np.maximum(x, 1e-30)) - np.log(lo_m)) / (np.log(hi_m) - np.log(lo_m)) - 1
        def mom(w):
            den = (w * wv).sum()
            return np.array([(w * wv * cheb(u, n)).sum() / den for n in range(1, nmax + 1)])
        mp, mq = mom(ev["weight"]), mom(res.weights)
        print(f"  {key}:")
        for n in range(nmax):
            print(f"     T_{n+1}: prior {mp[n]:+.4f}   MaxEnt {mq[n]:+.4f}   FO {fo[n]:+.4f}"
                  f"   |MaxEnt-FO| {abs(mq[n]-fo[n]):.4f}  (prior gap {abs(mp[n]-fo[n]):.4f})")

    # -------- uncertainty variants (convention of phistar_prediction.py /
    # fig_dy_spectra.py): the MaxEnt curve carries a translucent fill = per-bin
    # envelope over the 6 warm-started scale re-solves of the weights, and stat
    # bars = bootstrap-over-seeds spread (+) the sample's own per-bin MC error.
    # The fixed-order reference carries its 7-point scale envelope (+) seed
    # scatter, computed PER SEED on the final binning.  The moment construction
    # below mirrors the central call above VERBATIM (same maps, same tags) --
    # only scale_idx / the seed list change.
    import functools
    import nnlojet_moments as _nm
    if not isinstance(_nm._load, functools._lru_cache_wrapper):
        _nm._load = functools.lru_cache(maxsize=None)(_nm._load)
    sd2s = set(sd2) if mixed else set()

    def build_M(seed_list, scale_idx=0):
        seed_list = [int(s) for s in seed_list]
        Ms = fo_moments_smooth_from_nnlojet(
            ZDIR, RUN, CH, seed_list,
            born_tags={"mll": "mll", "y_abs": "absyz"},
            n_born=6, n_recoil=12, x_match=XM, x_hi=XHI, soft_lo=SOFT,
            recoil_cfg_name="ptj1", norm_born="norm_born",
            w0="prof_wj1_0", wtag="prof_wj1", prefix=PREFIX, scale_idx=scale_idx)
        if mixed:
            s2 = [s for s in seed_list if s in sd2s] or list(sd2)
            add_profiled_recoil(Ms, ZDIR, RUN, CH, s2, "ptj2", wtag="prof_wj2",
                                w0="prof_wj2_0", n_recoil=6, x_match=XM, x_hi=XHI,
                                soft_lo=SOFT, prefix=PREFIX, scale_idx=scale_idx)
            add_mixed_moments(Ms, ZDIR, RUN, CH, s2, "ptj12", wtag="prof_wj12",
                              w0="prof_wj12_0", n_max=NMIX, prefix=PREFIX,
                              scale_idx=scale_idx)
        return Ms

    lam0 = res.report["lam"]
    scale_w, boot_w = [], []
    print("\nuncertainty variants: 6 scale re-solves (warm-started) ...", flush=True)
    for s_ in range(1, 7):
        try:
            scale_w.append(upgrade(ev, build_M(seeds, s_), {**cfg, "lam0": lam0}).weights)
        except Exception as e_:
            print(f"  scale {s_} re-solve FAILED: {e_}")
    NBOOT = 20
    print(f"uncertainty variants: {NBOOT} seed-bootstrap re-solves ...", flush=True)
    rng = np.random.default_rng(20260816)
    for b_ in range(NBOOT):
        try:
            boot_w.append(upgrade(ev, build_M(rng.choice(seeds, len(seeds), replace=True)),
                                  {**cfg, "lam0": lam0}).weights)
        except Exception as e_:
            print(f"  bootstrap {b_} re-solve FAILED: {e_}")
    print(f"  variants ready: {len(scale_w)}/6 scale, {len(boot_w)}/{NBOOT} bootstrap")

    def fo_band_members(tag, e):
        """7-point scale members + per-seed scatter of the FO reference on the
        final edges e, oriented onto the analysis observable's axis."""
        mirrored = tag in MIRRORED_TAGS
        ee = (np.pi - np.asarray(e, float))[::-1] if mirrored else e
        fb = fo_curve_band(ZDIR, RUN, CH, seeds, tag, edges=ee, prefix=PREFIX,
                           members=True)
        if fb is None:
            return None
        _, _, cen, _, _, fst, mem = fb
        mem = np.asarray(mem)
        if mirrored:
            cen, fst, mem = cen[::-1], fst[::-1], mem[:, ::-1]
        return cen, fst, mem

    # ---------------- figure ----------------
    def native_edges(tag, mirror=False):
        """The fixed-order histogram's own edges.  It is the ratio DENOMINATOR:
        rebinning it coarse->fine makes a staircase and the ratio alternates."""
        import glob as _g
        f = sorted(_g.glob(ZDIR + f"/**/*.{tag}.s*.dat", recursive=True))[0]
        rows = [l.split() for l in open(f) if l.strip() and not l.startswith("#")]
        lo = np.array([float(r[0]) for r in rows]); hi = np.array([float(r[2]) for r in rows])
        e_ = np.concatenate([lo[:1], hi])
        if mirror: e_ = (np.pi - e_)[::-1]
        return e_[e_ > 0]
    panels = [("ptj1", native_edges("ptj1_a"), r"$p_T^{j_1}$ [GeV]", True,  "constrained", "ptj1_a"),
              ("ptj2", native_edges("ptj2_a"), r"$p_T^{j_2}$ [GeV]", True,  "predicted", "ptj2_a"),
              ("pimdphi", native_edges("dphil_a", mirror=True), r"$\pi-\Delta\phi_{\ell\ell}$", True, "predicted", "dphil_a")]

    def fo_ref(tag):
        """channel-summed FO reference distribution (lo, hi, density)."""
        lo = hi = None; tot = None
        for s_ in seeds:
            for ch in CH:
                r0 = _load(os.path.join(ZDIR, f"ch_{ch}", f"{PREFIX}.{RUN}.{ch}.{tag}.s{s_}.dat"))
                if r0 is None: continue
                lo, _, hi, v, _ = r0
                tot = v[:, 0].copy() if tot is None else tot + v[:, 0]
        return (lo, hi, tot) if tot is not None else None
    # ONE observable per figure.
    for j, (key, e, lab, logx, role, fotag) in enumerate(panels):
        fig, ax = plt.subplots(2, 1, figsize=(7.0, 7.8), squeeze=False,
                               gridspec_kw={"height_ratios": [2.1, 1.15], "hspace": 0.07})
        a_, r_ = ax[0, 0], ax[1, 0]
        # pT_j2 exists only for >=2-jet events; the FO histogram contains only
        # those, so normalise the prior/MaxEnt over the same subset.
        sub = (ev["ptj2"] > 0) if key == "ptj2" else np.ones(len(ev[key]), bool)
        def d(w, sub=sub):
            ww = w[sub]
            return np.histogram(ev[key][sub], e, weights=ww / ww.sum())[0] / np.diff(e)
        hp, hq = d(ev["weight"]), d(res.weights)
        ctr_ = np.sqrt(e[:-1] * e[1:])
        # FIXED-ORDER reference, and the ratio denominator (no data loaded for DY+jet)
        fo_i = None; fo_valid = None; fo_band = None; fo_stat = None
        fc = fo_ref(fotag)
        if fc is not None:
            flo, fhi, fv = fc
            if key == "pimdphi":
                # NNLOJET books dphi_l1l2 (= dphi); we plot pi - dphi.  Mirror it.
                flo, fhi, fv = (np.pi - fhi)[::-1], (np.pi - flo)[::-1], fv[::-1]
                keep = fhi > flo
                flo, fhi, fv = flo[keep], fhi[keep], fv[keep]
            good = (fv > 0) & (fhi > flo)
            if good.sum() > 2:
                # SAME EDGES as prior and MaxEnt -- and the fixed order is the
                # ratio DENOMINATOR here, so interpolating it would contaminate
                # every other curve's ratio.
                fo = rebin_density(flo[good], fhi[good], fv[good], e)
                gd = np.isfinite(fo) & (fo > 0)
                yy = np.where(gd, fo, np.nan)
                yy = yy * ((hq * np.diff(e)).sum() / np.nansum(yy * np.diff(e)))
                # Fixed order is a PREDICTION only above the seam.  For pi-dphi
                # the seam maps over as pi-dphi ~ pT/pT_lep ~ XM/45; below it the
                # unresummed Sudakov logarithms flatten the curve and it means
                # nothing, so draw it faded rather than let it read as a target.
                xs_ = XM / 45.0 if key == "pimdphi" else XM
                ab = ctr_ >= xs_
                a_.stairs(np.where(ab, yy, np.nan), e, color=C["fo"], ls=":", lw=LW["fo"],
                          label=(r"fixed order (NLO $Z$+jet)" if key == "ptj1"
                                 else r"fixed order (LO $Z$+2 jets)"))
                a_.stairs(np.where(~ab, yy, np.nan), e, color=C["fo"], ls=":",
                          lw=1.4, alpha=FADE)
                fo_i = yy
                fo_valid = ab
                # FO band: SHAPE-ONLY 7-point scale envelope (each scale member
                # normalized to the SAME integral the central is anchored to,
                # BEFORE the envelope -- this panel shows normalized densities,
                # on which the sample has no rate freedom) (+) per-seed scatter
                # in quadrature.
                fbm = fo_band_members(fotag, e)
                if fbm is not None:
                    cen_b, fst_b, mem_b = fbm
                    bwv = np.diff(e)
                    tgt = (hq * bwv).sum()      # the central curve's anchor integral
                    mem_n = []
                    for c_ in mem_b:
                        c_ = np.where(gd, c_, np.nan)
                        I_ = np.nansum(c_ * bwv)
                        mem_n.append(c_ * (tgt / I_) if I_ > 0
                                     else np.full(len(bwv), np.nan))
                    mem_n = np.array(mem_n)
                    lo_b = np.minimum(yy, np.nanmin(mem_n, 0))
                    hi_b = np.maximum(yy, np.nanmax(mem_n, 0))
                    I_c = np.nansum(np.where(gd, cen_b, np.nan) * bwv)
                    st_b = (fst_b * (tgt / I_c) if I_c > 0
                            else np.full(len(bwv), np.nan))
                    fo_band = 0.5 * (hi_b - lo_b)   # BAND = scale
                    fo_stat = st_b                  # CANDLE = seed scatter
        ref = np.where(np.isfinite(fo_i), fo_i, np.nan) if fo_i is not None else np.maximum(hq, 1e-30)
        a_.stairs(hp, e, color=C["prior"], ls="--", lw=2.0, label=r"PS+LO prior")
        a_.stairs(hq, e, color=C["maxent"], lw=3.0, label=r"MaxEnt ($0\%\ w<0$)")
        hq2 = d(res_dphi.weights) if (key == "pimdphi" and res_dphi is not None) else None
        if hq2 is not None:
            a_.stairs(hq2, e, color=C_DPHI, lw=2.4, ls="-.",
                      label=rf"MaxEnt $+\,\Delta\phi$ tower ($n\le{N_DPHI}$)")
        # The ratio denominator IS the fixed order, so below the seam we would be
        # dividing by the unresummed Sudakov region.  Fade the ratio there too,
        # rather than draw a solid line against a denominator we do not trust.
        vok = fo_valid if fo_valid is not None else np.ones(len(e) - 1, bool)
        for h_, ckey, lw_ in ((hp, "prior", 1.8), (hq, "maxent", 2.4)):
            r_.stairs(np.where(vok, h_ / ref, np.nan), e, color=C[ckey],
                      ls=LS[ckey], lw=lw_)
            r_.stairs(np.where(~vok, h_ / ref, np.nan), e, color=C[ckey],
                      ls=LS[ckey], lw=lw_ * 0.7, alpha=FADE)
        if hq2 is not None:
            r_.stairs(np.where(vok, hq2 / ref, np.nan), e, color=C_DPHI, ls="-.", lw=2.2)
            m2_ = vok & np.isfinite(hq2) & (hq2 > 0) & np.isfinite(ref)
            print(f"  pimdphi above seam: median|r-1| prior {100*np.median(np.abs(hp[m2_]/ref[m2_]-1)):.1f}%  "
                  f"MaxEnt {100*np.median(np.abs(hq[m2_]/ref[m2_]-1)):.1f}%  "
                  f"+dphi tower {100*np.median(np.abs(hq2[m2_]/ref[m2_]-1)):.1f}%  "
                  f"(MaxEnt/FO range {np.nanmin(hq[m2_]/ref[m2_]):.2f}..{np.nanmax(hq[m2_]/ref[m2_]):.2f})")
        r_.axhline(1, color="k", lw=0.8)
        # ---- uncertainty visuals, one style per role (shared convention) ----
        # FO denominator: gray band around 1 = scale envelope (+) seed scatter.
        # MaxEnt: translucent fill = envelope over the 6 scale re-solves; bars =
        # seed-bootstrap spread (+) own MC error.  Prior: own MC bars.  Bars are
        # staggered per series so they stay legible.
        if fo_i is not None and fo_band is not None:
            tot_ = fo_band / np.where(np.isfinite(ref) & (ref > 0), ref, np.inf)
            r_.fill_between(ctr_, np.where(vok, 1 - tot_, np.nan),
                            np.where(vok, 1 + tot_, np.nan),
                            step="mid", color=C["band"], alpha=0.55, lw=0)
            if fo_stat is not None:
                r_.errorbar(stagger(e, 0, 3), np.where(vok, np.ones_like(ctr_), np.nan),
                            yerr=np.where(vok, fo_stat / np.where(
                                np.isfinite(ref) & (ref > 0), ref, np.inf), np.nan),
                            fmt="none", ecolor="0.35", elinewidth=1.0, capsize=1.6)
        # statistics FIRST: the MaxEnt band below is the sample's TOTAL
        # uncertainty, like for like with the reference band above, which is
        # that reference's scale envelope combined with its own scatter.  A
        # scale-only band drawn against a scale-plus-statistics one understates
        # the upgrade several-fold.
        boot_sd = (np.array([d(wv) for wv in boot_w]).std(0, ddof=1)
                   if len(boot_w) > 1 else np.zeros(len(e) - 1))

        def mc_of(wv):
            ww_ = wv[sub]; qn_ = ww_ / ww_.sum()
            s2_, _ = np.histogram(ev[key][sub], e, weights=qn_ * qn_)
            return np.sqrt(s2_) / np.diff(e)

        q_err = np.hypot(boot_sd, mc_of(res.weights)); p_err = mc_of(ev["weight"])
        q_lo = q_hi = None
        q_scale_half = None
        if scale_w:
            hs_ = np.array([d(wv) for wv in scale_w])
            q_lo = np.minimum(hq, hs_.min(0)); q_hi = np.maximum(hq, hs_.max(0))
            q_scale_half = 0.5 * (q_hi - q_lo)   # BAND stays the scale envelope
            r_.fill_between(ctr_, np.where(vok, q_lo / ref, np.nan),
                            np.where(vok, q_hi / ref, np.nan),
                            step="mid", color=C["maxent"], alpha=0.20, lw=0)
        r_.errorbar(stagger(e, 1, 3), np.where(vok, hp / ref, np.nan),
                    yerr=np.where(vok, p_err / ref, np.nan), fmt="none",
                    ecolor=C["prior"], elinewidth=1.1, capsize=1.8)
        r_.errorbar(stagger(e, 2, 3), np.where(vok, hq / ref, np.nan),
                    yerr=np.where(vok, q_err / ref, np.nan), fmt="none",
                    ecolor=C["maxent"], elinewidth=1.3, capsize=2.2)
        okb = vok & np.isfinite(hq) & (hq > 0)
        halves = []
        if q_scale_half is not None:
            halves.append("MaxEnt scale +-%.2f%%" % (100 * np.nanmedian(
                (q_scale_half / hq)[okb])))
        halves.append("MaxEnt stat +-%.2f%%" % (100 * np.nanmedian(
            np.where(okb, q_err / hq, np.nan))))
        if fo_band is not None:
            fo_ok = okb & np.isfinite(fo_i) & (fo_i > 0)
            halves.append("FO +-%.2f%%" % (100 * np.nanmedian(
                np.where(fo_ok, fo_band / fo_i, np.nan))))
        print(f"  {key} band half-widths (median, % of central): " + "  ".join(halves))
        # Where the reweighted effective statistics collapse, the prior simply
        # has no events: pT_j2's hard tail needs a Z+2-jet matrix element, which
        # a Z+1-jet-plus-shower prior cannot supply, and reweighting cannot
        # create events.  Grey those bins rather than let them read as failure.
        bad, _eff = support_mask(ev[key], res.weights, e, min_eff=100.0)
        shade_unsupported((a_, r_), e, bad)
        for p in (a_, r_):
            if key == "ptj1":
                p.axvspan(XM, min(XHI, e[-1]), color="#ffd24d", alpha=0.13)
            p.axvline(XM / 45.0 if key == "pimdphi" else XM,
                      color=C["seam"], lw=2.0, ls="--")
        if logx: a_.set_xscale("log"); r_.set_xscale("log")
        a_.set_yscale("log"); a_.tick_params(labelbottom=False)
        r_.set_xlabel(lab); r_.set_ylim(0.5, 1.6)
        a_.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}X$")
        r_.set_ylabel(r"ratio to fixed order")
        a_.legend(loc="lower left", fontsize=12)
        constrained = key in ("ptj1",) or (bool(mixed) and key == "ptj2")
        role_now = "constrained" if constrained else "predicted (never constrained)"
        extra = (r", with the mixed $\langle T_m(p_T^{j_1})T_n(p_T^{j_2})\rangle$"
                 if bool(mixed) and key in ("ptj1", "ptj2") else "")
        a_.set_title(rf"{lab}, {role_now}")
        out = os.path.join(HERE, f"fig_zj_{key}.pdf")
        fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
