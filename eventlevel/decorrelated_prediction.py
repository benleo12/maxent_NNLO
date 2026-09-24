#!/usr/bin/env python3
r"""Predictions for lepton-angle observables nearly uncorrelated with the
constrained recoil (JHEP, transport section), drawn with the uncertainty
conventions of every other Drell-Yan figure: band = seven-point scale
envelope, bars = statistics (MaxEnt: seed bootstrap (+) own MC error;
generators and prior: own MC error).

The observables (|delta eta_ll|, |eta_lead|, |cos theta*_CS|) need the lepton
four-vectors, which the fiducial samples of the other figures (v3/v4 npz) do
not keep.  They are rebuilt here from the showered PARENTS of those samples in
exactly the order the samples were built (fiducial mask of
make_dy_atlas_npz_born.py on the concatenated parents), and the lengths are
checked, so the per-event weights, the stored seven-point scale weights and
the MaxEnt band variants of dy_band_weights.npz all apply unchanged.
"""
import os
import sys
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import use_pub_style, C, gen_scale_weights
use_pub_style(base=17)
from bandviz import norm_dens, mc_err, stagger, draw_series_unc
from maxent_upgrade import upgrade
from nnlojet_moments import fo_moments_smooth_from_nnlojet, common_seeds

BASE = os.path.join(os.environ.get("NNLOJET_ROOT",
        os.path.expanduser("~/nnlojet-v1.0.2")), "dy_profile_log30_hi")
XM, XHI, SOFT, XMAP = 30.0, 500.0, 30.0, 2500.0
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]

# fiducial sample of the other DY figures -> its showered parents (build order)
# Showered parent files.  These are large and live outside the repo; set
# DY_PARENTS to wherever they were unpacked.  Only the parent-order rebuild
# needs them -- every other script reads the small *_atlas_*.npz in this dir.
PARENTS = os.environ.get("DY_PARENTS", os.path.expanduser("~/dy_workdir"))

SAMPLES = {
    "prior":   ("dy_prior_atlas_v3.npz",   ["dy_psLO_born_1.npz"]),
    "minnlo":  ("dy_minnlo_atlas_v4.npz",
                [f"{PARENTS}/minnlo_matched/dy_minnlo_matched_s{i}.npz" for i in range(1, 17)]),
    "mcatnlo": ("dy_mcatnlo_atlas_v4.npz", ["dy_mcatnlo_born_sh.npz", "dy_mcatnlo_born_ext_sh.npz"]),
    "powheg":  ("dy_powheg_atlas_v6.npz",
                [f"{PARENTS}/powheg_gmu/dy_powheg_gmu_s{i}.npz" for i in range(1, 9)]),
}
GENS = [("minnlo", r"MiNNLO$_{\mathrm{PS}}$", 23), ("mcatnlo", "MC@NLO", 5), ("powheg", "POWHEG", 1)]
OBS = [("deta", r"$|\Delta\eta_{\ell\ell}|$", np.linspace(0, 4.0, 26)),
       ("eta_lead", r"$|\eta_{\ell,\mathrm{lead}}|$", np.linspace(0, 2.5, 26)),
       ("cts", r"$|\cos\theta^*_{\mathrm{CS}}|$", np.linspace(0, 1.0, 26))]


def _pt(v):
    return np.hypot(v[:, 0], v[:, 1])


def _eta(v):
    pm = np.sqrt((v[:, 0:3] ** 2).sum(1))
    return np.arctanh(np.clip(v[:, 2] / np.maximum(pm, 1e-30), -1 + 1e-12, 1 - 1e-12))


def fid_mask(lpb, lmb, lp, lm):
    """EXACTLY the mask of make_dy_atlas_npz_born.py (kin() on the Born
    leptons, plus its `good` requirement on both lepton sets)."""
    s = lpb + lmb
    mll = np.sqrt(np.maximum(s[:, 3] ** 2 - (s[:, 0:3] ** 2).sum(1), 0.0))
    fid = ((mll >= 66) & (mll <= 116) & (_pt(lpb) > 27) & (_pt(lmb) > 27)
           & (np.abs(_eta(lpb)) < 2.5) & (np.abs(_eta(lmb)) < 2.5))
    good = (lpb[:, 3] > 0) & (lp[:, 3] > 0)
    return fid & good


def angles(lp, lm):
    s = lp + lm
    mll = np.sqrt(np.maximum(s[:, 3] ** 2 - (s[:, 0:3] ** 2).sum(1), 0.0))
    pT = np.hypot(s[:, 0], s[:, 1])
    ep, em = _eta(lp), _eta(lm)
    p1p = (lp[:, 3] + lp[:, 2]) / np.sqrt(2); p1m = (lp[:, 3] - lp[:, 2]) / np.sqrt(2)
    p2p = (lm[:, 3] + lm[:, 2]) / np.sqrt(2); p2m = (lm[:, 3] - lm[:, 2]) / np.sqrt(2)
    pz = lp[:, 2] + lm[:, 2]
    cs = np.abs(np.sign(pz) * 2 * (p1p * p2m - p1m * p2p)
                / np.maximum(mll * np.sqrt(mll ** 2 + pT ** 2), 1e-12))
    return dict(deta=np.abs(ep - em), eta_lead=np.where(_pt(lp) >= _pt(lm), np.abs(ep), np.abs(em)),
                cts=np.clip(cs, 0, 1), pT_ll=pT)


def rebuild(parents, n_expect):
    """Angular observables of the fiducial sample, in the sample's order."""
    parts = [np.load(os.path.join(HERE, f), allow_pickle=True) for f in parents]
    arr = {k: [np.asarray(p[k], float) for p in parts]
           for k in ("l_plus_born", "l_minus_born", "l_plus", "l_minus")}
    for perm in itertools.permutations(range(len(parts))):
        cat = {k: np.concatenate([v[i] for i in perm]) for k, v in arr.items()}
        m = fid_mask(cat["l_plus_born"], cat["l_minus_born"], cat["l_plus"], cat["l_minus"])
        if int(m.sum()) == n_expect:
            return angles(cat["l_plus_born"][m], cat["l_minus_born"][m]), [parents[i] for i in perm]
    raise RuntimeError(f"no parent order reproduces the fiducial sample length {n_expect:,} "
                       f"for {parents}")


def spearman(x, y, n=300_000, seed=1):
    idx = np.random.default_rng(seed).choice(len(x), min(n, len(x)), replace=False)
    rx = np.argsort(np.argsort(x[idx])).astype(float)
    ry = np.argsort(np.argsort(y[idx])).astype(float)
    return abs(float(np.corrcoef(rx, ry)[0, 1]))


def main():
    # ---- prior: the same 1M rng(0) subsample and band variants as every DY figure
    P = dict(np.load(os.path.join(HERE, SAMPLES["prior"][0])))
    n = len(P["w"])
    ang, order = rebuild(SAMPLES["prior"][1], n)
    print(f"prior: {n:,} fiducial events rebuilt from {order}")
    idx = np.random.default_rng(0).choice(n, min(1_000_000, n), replace=False)
    ev = dict(mll=P["mll"][idx].astype(float), y_abs=np.abs(P["y_ll"][idx]).astype(float),
              pT_ll=P["pT_ll"][idx].astype(float), phistar=P["phistar"][idx].astype(float),
              weight=P["w"][idx].astype(float))
    for k in ("deta", "eta_lead", "cts"):
        ev[k] = ang[k][idx]
    print("  |rho_S(., pT_ll)| on the prior: " + "  ".join(
        f"{k}={spearman(ev[k], ev['pT_ll']):.3f}" for k in ("deta", "eta_lead", "cts")))
    M = fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, common_seeds(BASE, "DY_MOMENTS", CH6),
                                       born_tags={"mll": "mll", "y_abs": "absyz"},
                                       n_born=12, n_recoil=20, x_match=XM, x_hi=XMAP, soft_lo=SOFT)
    cfg = dict(born={"mll": {"range": (66., 116.), "map": "bw"},
                     "y_abs": {"range": (0., 2.4), "map": "lin"}},
               recoil={"pT_ll": {"range": (SOFT, XMAP), "map": "log", "soft_lo": SOFT,
                                 "profile": {"a": XM, "b": 2 * XM, "c": XHI}}},
               followers=["deta", "eta_lead", "cts"])
    print("solving ...", flush=True)
    res = upgrade(ev, M, cfg)
    print(f"  effN={100*res.effN:.1f}%  closure={res.closure:.1e}  neg-wt={100*np.mean(res.weights<=0):.0f}%")
    mx_ws = mx_boot_w = None
    bwf = os.path.join(HERE, "dy_band_weights.npz")
    if os.path.exists(bwf):
        BW = dict(np.load(bwf))
        if np.allclose(BW["central"], res.weights, rtol=1e-6, atol=0):
            mx_ws = np.column_stack([res.weights] + list(BW["scale_w"])); mx_boot_w = BW["boot_w"]
        else:
            print("  WARNING: dy_band_weights.npz stale; MaxEnt drawn without bands")

    # ---- generators: the paper's fiducial samples, their weights and 7-point scale weights
    gens = {}
    for key, lbl, neg in GENS:
        fn, parents = SAMPLES[key]
        G = dict(np.load(os.path.join(HERE, fn)))
        try:
            a_, order = rebuild(parents, len(G["w"]))
        except RuntimeError as err:
            print(f"  {lbl}: {err} -- skipped"); continue
        gens[key] = dict(lbl=lbl, neg=neg, w=G["w"].astype(float),
                         ws=(gen_scale_weights(G) if "w_scale" in G else None), **a_)
        print(f"  {lbl}: {len(G['w']):,} fiducial events from {order}")

    fig, ax = plt.subplots(2, len(OBS), figsize=(5.6 * len(OBS), 7.2), squeeze=False,
                           gridspec_kw={"height_ratios": [2.1, 1.15], "hspace": 0.07, "wspace": 0.28})
    nser = 2 + len(gens)
    for j, (key, lab, e) in enumerate(OBS):
        a, r = ax[0, j], ax[1, j]
        hp = norm_dens(ev[key], ev["weight"], e)
        hq = norm_dens(ev[key], res.weights, e)
        ref = np.maximum(hp, 1e-30)
        a.stairs(hp, e, color=C["prior"], ls="--", lw=2.0, label=r"PS+LO prior")
        a.stairs(hq, e, color=C["maxent"], lw=3.0, label=r"MaxEnt ($0\%\ w<0$)")
        r.stairs(hq / ref, e, color=C["maxent"], lw=2.4)
        # prior: its own MC bars at one; MaxEnt: scale band (+) bootstrap-and-MC bars
        r.errorbar(stagger(e, 0, nser), np.ones(len(e) - 1), yerr=mc_err(ev[key], ev["weight"], e) / ref,
                   fmt="none", ecolor=C["prior"], elinewidth=1.1, capsize=1.8)
        boot = (np.array([norm_dens(ev[key], w_, e) for w_ in mx_boot_w]).std(0, ddof=1)
                if mx_boot_w is not None else None)
        draw_series_unc(r, ev[key], res.weights, mx_ws, e, ref, C["maxent"], 1, nser, boot=boot, alpha=0.18)
        for k, (gk, g) in enumerate(gens.items()):
            hg = norm_dens(g[key], g["w"], e)
            a.stairs(hg, e, color=C[gk], lw=1.9, label=rf"{g['lbl']} (${g['neg']}\%\ w<0$)")
            r.stairs(hg / ref, e, color=C[gk], lw=1.8)
            draw_series_unc(r, g[key], g["w"], g["ws"], e, ref, C[gk], 2 + k, nser, alpha=0.13)
            ok = (hg > 0) & (hq > 0) & (hp > 0)
            print(f"  {key:9s} vs {g['lbl']:22s}: median|MaxEnt/gen-1| = {100*np.median(np.abs(hq[ok]/hg[ok]-1)):.1f}%   "
                  f"median|prior/gen-1| = {100*np.median(np.abs(hp[ok]/hg[ok]-1)):.1f}%")
        r.axhline(1, color="k", lw=0.8)
        r.set_ylim(0.80, 1.30); r.set_xlabel(lab)
        a.tick_params(labelbottom=False)
        a.set_title(rf"{lab}, predicted")
        if j == 0:
            a.set_ylabel(r"$(1/\sigma)\,\mathrm{d}\sigma/\mathrm{d}X$")
            r.set_ylabel(r"ratio to PS+LO prior")
            a.legend(loc="upper right", labelspacing=0.3, fontsize=12)
    out = os.path.join(HERE, "fig_decorrelated_prediction.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
