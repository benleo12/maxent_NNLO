#!/usr/bin/env python3
r"""Diphoton with EVENT-LEVEL moments vs ATLAS 1704.03839, publication style.

Three constrained observables -- the Born pair (m_aa, |cos theta*|) and the
recoil (pT_aa) -- and two pure predictions (Delta phi, a_T).

|cos theta*| is constrained because it must be: the photon pT cuts lock it to
m_aa, so mass moments alone move it the wrong way (17.9% vs ATLAS) while adding
it recovers 4.7% AND relieves m_aa from 11.0% to 9.8%.  Delta phi is left free
on purpose -- it is a recoil observable and cannot be a Born constraint (see
aa_angular_test.py) -- yet it improves from 31.2% to 7.0% as a consequence.
(40 channel- and tag-complete NNLOJET seeds.)

The shaded band below dphi = 2 is outside the prior's support and says nothing
about the method: the prior is a 2->2 matrix element plus shower, so that region
is reachable only by ISR recoil -- 1 event in 14440 even at pTHat in [200,400),
with ISR already on -- and reweighting cannot create events.  The boundary is
set by effective statistics, not by eye: N_eff per bin group runs 3.8, 15.2,
748 across [0,1), [1,1.5), [1.5,2) and then jumps to 9700 above 2.  Populating it
needs a gamma-gamma+jet (2->3) matrix element, not more statistics.

Separately, the excursion just above the band near dphi = 2.2 rad is real and
unresolved: the prior is ~4x below data there and the reweighting overcorrects.
Constraining pi-dphi as a profiled RECOIL observable (its correct slot; as a
Born constraint the dual is infeasible) reduces the local peak from 83% to
23-34% but degrades agreement with ATLAS overall at N = 2, 3 and 6 moments,
because the fixed-order pi-dphi moments are ten times noisier than the
|cos theta*| ones.  That is a statistics limitation of the fixed-order input,
not of the method, so dphi is shipped unconstrained.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import use_pub_style, C, LW, rebin_density
use_pub_style(base=17)
from aa_vs_data import load_prior_full, dens
from nnlojet_moments import _load, common_seeds

D = dict(np.load(os.path.join(HERE, "atlas_aa_8tev.npz"), allow_pickle=True))
W = dict(np.load(os.path.join(HERE, "aa_eventlevel_weights.npz")))

GGDIR = "/Users/user/nnlojet-v1.0.2/gg_moments"
DPHI_SUPP = 2.0   # below this the reweighted effective statistics fall under
                  # 1000 per bin group (3.8 / 15.2 / 748 in [0,1)/[1,1.5)/[1.5,2)
                  # against 9700 just above): the prior has no support there.
FOTAG = {"m_aa": "m_aa", "pt_aa": "pt_aa", "costh_aa": "cosCS",
         "dphi_aa": "dphi_aa", "at_aa": "at_aa"}


def fo_ref(tag):
    seeds = common_seeds(GGDIR, "GG_MOMENTS", ["LO","R","V","RR","RV","VV"], prefix="GG")
    lo = hi = None; tot = None
    for s_ in seeds:
        for ch in ("LO","R","V","RR","RV","VV"):
            r0 = _load(os.path.join(GGDIR, f"ch_{ch}", f"GG.GG_MOMENTS.{ch}.{tag}.s{s_}.dat"))
            if r0 is None: continue
            lo, _, hi, v, _ = r0
            tot = v[:, 0].copy() if tot is None else tot + v[:, 0]
    return (lo, hi, tot) if tot is not None else None


PANELS = [
    ("m_aa",     "m_aa",     r"$m_{\gamma\gamma}$ [GeV]",            True,  "constrained"),
    ("pt_aa",    "pt_aa",    r"$p_T^{\gamma\gamma}$ [GeV]",          True,  "constrained"),
    ("costh_aa", "costh_aa", r"$|\cos\theta^*|$",                    False, "constrained"),
    ("dphi_aa",  "dphi_aa",  r"$\Delta\phi_{\gamma\gamma}$",         False, "predicted"),
    ("at_aa",    "at_aa",    r"$a_T^{\gamma\gamma}$ [GeV]",          True,  "predicted"),
]

# Accuracy is a property of the OBSERVABLE, not of the calculation.  From the
# inclusive gamma-gamma calculation at NNLO, the Born observables (mass, the
# Collins-Soper angle, the rapidity) are NNLO; anything that requires a real
# emission to be non-zero -- the recoil pT, a_T, and dphi away from pi -- is only
# NLO(gamma-gamma+jet) accurate.  Label each curve by what it actually is.
FO_ORDER = {"m_aa": r"NNLO", "costh_aa": r"NNLO", "y_aa": r"NNLO",
            "pt_aa": r"NLO $\gamma\gamma$+jet",
            "dphi_aa": r"NLO $\gamma\gamma$+jet",
            "at_aa": r"NLO $\gamma\gamma$+jet"}


def main():
    ev = load_prior_full()
    ev = {k: v[W["idx"]] for k, v in ev.items()}
    wpr, wnw = ev["weight"], W["weights"]

    # ONE observable per figure, as everywhere else.
    for j, (key, dk, lab, logx, role) in enumerate(PANELS):
        fig, ax = plt.subplots(2, 1, figsize=(6.6, 7.6), squeeze=False,
                               gridspec_kw={"height_ratios": [2.1, 1.15], "hspace": 0.07})
        a, r = ax[0, 0], ax[1, 0]
        lo = np.asarray(D[f"{dk}_lo"], float); hi = np.asarray(D[f"{dk}_hi"], float)
        val = np.asarray(D[f"{dk}_val"], float); err = np.asarray(D[f"{dk}_err"], float)
        e = np.concatenate([lo[:1], hi]); ctr = 0.5 * (lo + hi)
        hp, hq = dens(ev[key], wpr, e), dens(ev[key], wnw, e)
        bw = np.diff(e)
        norm = lambda h: h / (h * bw).sum()
        dv = norm(val); pp = norm(hp); qq = norm(hq)
        m = (dv > 0) & (hp > 0)
        med = lambda h: 100 * np.median(np.abs(h[m] / dv[m] - 1))

        a.errorbar(ctr[m], dv[m], yerr=(err / (val * bw).sum())[m], fmt="o", color="k",
                   ms=4, lw=1.1, label=r"ATLAS 1704.03839", zorder=9)
        a.stairs(np.where(m, pp, np.nan), e, color=C["prior"], ls="--", lw=2.0,
                 label=rf"PS+LO prior ({med(pp):.0f}\%)")
        a.stairs(np.where(m, qq, np.nan), e, color=C["maxent"], lw=3.0,
                 label=rf"MaxEnt ({med(qq):.0f}\%)")
        # FIXED ORDER, normalised over the plotted range
        fc = fo_ref(FOTAG.get(key, key))
        if fc is not None:
            flo, fhi, fv = fc
            if key == "dphi_aa":
                # NNLOJET books pi_dphi_g1g2 (= pi - dphi) under this name; mirror
                # it onto the data's dphi axis before comparing.
                flo, fhi, fv = (np.pi - fhi)[::-1], (np.pi - flo)[::-1], fv[::-1]
            g = (fv > 0) & (fhi > flo)
            if g.sum() > 2:
                # SAME EDGES as every other line on this panel: integrate the
                # fixed order onto the measurement binning instead of drawing it
                # on NNLOJET's native grid and interpolating the ratio.
                fo = rebin_density(flo[g], fhi[g], fv[g], e)
                good = np.isfinite(fo) & (fo > 0)
                if good.sum() > 2:
                    fn = np.where(good, fo, np.nan)
                    fn = fn / np.nansum(fn * bw)          # same normalisation as pp/qq/dv
                    a.stairs(fn, e, color=C["fo"], ls=":", lw=LW["fo"],
                             label=rf"fixed order ({FO_ORDER.get(key, '')})")
                    r.stairs(np.where(m, fn / dv, np.nan), e, color=C["fo"], ls=":", lw=1.9)
        if key in ("pt_aa", "at_aa"):
            for p_ in (a, r):
                p_.axvline(28.0, color=C["seam"], lw=2.0, ls="--")
        if key == "dphi_aa":
            # OUTSIDE PRIOR SUPPORT.  The prior is a 2->2 matrix element plus
            # shower, so dphi < 1 can only be reached by ISR recoil: even at
            # pTHat in [200,400) it is 1 event in 14440, and ISR is already on.
            # Reweighting cannot create events, so nothing here is a statement
            # about the method.  Reaching it needs a gamma-gamma+jet (2->3)
            # matrix element, not more statistics.
            wl = wnw[ev[key] < DPHI_SUPP]
            effl = float(wl.sum() ** 2 / max((wl ** 2).sum(), 1e-300))
            nlow = int((ev[key] < DPHI_SUPP).sum())
            for p_ in (a, r):
                p_.axvspan(e[0], DPHI_SUPP, color="0.5", alpha=0.16, zorder=0)
            # The excursion just ABOVE the shaded band, near dphi = 2.2 rad, is
            # a known limitation and not a support artefact -- N_eff ~ 1e4
            # there.  Constraining pi-dphi as a profiled recoil does reduce it
            # (peak |ratio-1| 83% -> 23-34%) but degrades agreement with ATLAS
            # overall at every moment count tried, because the fixed-order
            # pi-dphi moments carry errors 0.024-0.069 against 0.002-0.007 for
            # |cos theta*|.  It needs more fixed-order statistics.
            a.text(DPHI_SUPP * 0.97, 0.97,
                   rf"outside prior support ($N={nlow}$, $N_{{\rm eff}}={effl:.0f}$)",
                   transform=a.get_xaxis_transform(), rotation=90, ha="right", va="top",
                   fontsize=11, color=C["seam"])
        if logx:
            a.set_xscale("log"); r.set_xscale("log")
        a.set_yscale("log"); a.tick_params(labelbottom=False)
        r.axhline(1, color="k", lw=0.8)
        rel = (err / np.maximum(val, 1e-30))
        r.fill_between(ctr[m], (1 - rel)[m], (1 + rel)[m], color=C["band"], alpha=0.55, step="mid")
        r.stairs(np.where(m, pp / dv, np.nan), e, color=C["prior"], ls="--", lw=1.8)
        r.stairs(np.where(m, qq / dv, np.nan), e, color=C["maxent"], lw=2.4)
        r.set_ylim(0.3, 1.9); r.set_xlabel(lab)
        a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}X$")
        r.set_ylabel(r"ratio to data")
        a.legend(loc="lower left", fontsize=12)
        a.set_title(rf"{lab}, {role}")
        out = os.path.join(HERE, f"fig_aa_{key}.pdf")
        fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
