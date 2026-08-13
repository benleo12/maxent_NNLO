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
                             add_mixed_moments)

ZDIR = "/Users/user/nnlojet-v1.0.2/zj_moments"
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
    cfg = dict(born={"mll": {"range": (66., 116.), "map": "lin"},
                     "y_abs": {"range": (0., 2.4), "map": "lin"}},
               recoil=recoil, mixed=mixed,
               followers=["pimdphi"] + ([] if mixed else ["ptj2"]),
               moment_selection=False)
    print("solving ...", flush=True)
    res = upgrade(ev, M, cfg)
    print(f"  effN {100*res.effN:.1f}%  closure {res.closure:.2e}  "
          f"neg-wt {100*np.mean(res.weights<=0):.1f}%")

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

    # ---------------- figure ----------------
    panels = [("ptj1", np.geomspace(20, 500, 26), r"$p_T^{j_1}$ [GeV]", True,  "constrained", "ptj1_a"),
              ("ptj2", np.geomspace(20, 300, 22), r"$p_T^{j_2}$ [GeV]", True,  "predicted", "ptj2_a"),
              ("pimdphi", np.geomspace(0.02, 3.0, 22), r"$\pi-\Delta\phi_{\ell\ell}$", True, "predicted", "dphil_a")]

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
        fo_i = None; fo_valid = None
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
        ref = np.where(np.isfinite(fo_i), fo_i, np.nan) if fo_i is not None else np.maximum(hq, 1e-30)
        a_.stairs(hp, e, color=C["prior"], ls="--", lw=2.0, label=r"PS+LO prior")
        a_.stairs(hq, e, color=C["maxent"], lw=3.0, label=r"MaxEnt ($0\%\ w<0$)")
        # The ratio denominator IS the fixed order, so below the seam we would be
        # dividing by the unresummed Sudakov region.  Fade the ratio there too,
        # rather than draw a solid line against a denominator we do not trust.
        vok = fo_valid if fo_valid is not None else np.ones(len(e) - 1, bool)
        for h_, ckey, lw_ in ((hp, "prior", 1.8), (hq, "maxent", 2.4)):
            r_.stairs(np.where(vok, h_ / ref, np.nan), e, color=C[ckey],
                      ls=LS[ckey], lw=lw_)
            r_.stairs(np.where(~vok, h_ / ref, np.nan), e, color=C[ckey],
                      ls=LS[ckey], lw=lw_ * 0.7, alpha=FADE)
        r_.axhline(1, color="k", lw=0.8)
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
        a_.set_title(rf"Drell--Yan$+$jet at 13 TeV: {lab} \small({role_now}{extra})"
                     "\n" rf"\small {len(seeds)} seed(s), channels {'+'.join(CH)};  "
                     rf"effN $={100*res.effN:.0f}\%$, closure $={res.closure:.1e}$",
                     fontsize=14)
        out = os.path.join(HERE, f"fig_zj_{key}.pdf")
        fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
