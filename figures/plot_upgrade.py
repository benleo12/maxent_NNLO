#!/usr/bin/env python3
r"""Process-agnostic plotter for the unified maxent_upgrade result.

plot_upgrade(events, result, config, ...) draws, for every observable declared in
`config` (Born, recoil, followers) plus any extra follower arrays, the SAME
standard panel the DY/diphoton figures use, straight from one `upgrade()` result.

REFERENCE SET (house rule -- always shown for context):
  * PS+LO prior            grey dashed   -- ALWAYS
  * fixed order (FO)        black dotted  -- ALWAYS when an `fo` curve is supplied
  * data                    black points  -- when a `data` histogram is supplied
On top of the references:
  * MaxEnt (this upgrade)   red solid + scale(+rate)(+stat) band from result.band
  * matched generators      coloured      -- when `generators` are supplied

Roles are tagged ('recoil - composite window' with the window shaded, 'Born -
full range', 'follower - never constrained'); each panel has a ratio sub-panel
(to data if given, else to the MaxEnt central).  Consuming only the generic
UpgradeResult (weights, band, effN, closure, x_match) plus the event arrays, the
identical call works for DY, gamma-gamma, gg->H and dijet.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_PRIOR = "0.55"
C_FO = "k"
C_POST = "#d62728"
C_GEN = ["#1f77b4", "#2ca02c", "#9467bd"]
C_WIN = "#ffd24d"


def _dens(x, w, edges):
    bw = np.diff(edges)
    h, _ = np.histogram(x, bins=edges, weights=w / w.sum())
    return h / bw


def _statrel(x, w, edges):
    s, _ = np.histogram(x, bins=edges, weights=w)
    s2, _ = np.histogram(x, bins=edges, weights=w ** 2)
    return np.sqrt(np.maximum(s2, 0)) / np.maximum(np.abs(s), 1e-30)


def _edges(x, obs, config, data):
    if data and obs in data:
        D = data[obs]
        return np.concatenate([np.asarray(D["lo"])[:1], np.asarray(D["hi"])])
    is_log = (obs in config.get("recoil", {})) or \
             (config.get("born", {}).get(obs, {}).get("map") == "log") or obs == "phistar"
    lo, hi = np.percentile(x, 0.2), np.percentile(x, 99.8)
    if is_log:
        pos = x[x > 0]
        lo = max(lo, np.percentile(pos, 0.5) if pos.size else 1e-3)
        return np.geomspace(max(lo, 1e-3), hi, 30)
    return np.linspace(lo, hi, 30)


def _fo_density(F, edges, x, wpo):
    r"""FO reference as a density on `edges`, rescaled so its integral over the
    FO's own support matches the MaxEnt integral there (so the dotted FO overlays
    the MaxEnt where FO is defined -- the window for a recoil, the full range for
    a Born variable).  F is a histogram {lo,hi,val} (val = dsigma/dx) or events
    {x,w}.  Returns (centers, density) sampled at the FO's own binning."""
    if "lo" in F:
        fe = np.concatenate([np.asarray(F["lo"])[:1], np.asarray(F["hi"])])
        fv = np.asarray(F["val"], float)
    else:
        fe = edges
        fv = _dens(np.asarray(F["x"], float), np.asarray(F["w"], float), fe)
    fbw = np.diff(fe)
    fint = float((fv * fbw).sum())
    in_rng = (x >= fe[0]) & (x < fe[-1])
    me_int = float(wpo[in_rng].sum() / wpo.sum())
    fv = fv * (me_int / max(fint, 1e-30))
    fctr = np.sqrt(fe[:-1] * fe[1:]) if (fe > 0).all() else 0.5 * (fe[:-1] + fe[1:])
    return fe, fctr, fv


def plot_upgrade(events, result, config, weight_key="weight",
                 data=None, data_label="data", fo=None, generators=None,
                 labels=None, title=None, outfile="fig_upgrade.png",
                 log_obs=None):
    born = list(config.get("born", {}).keys())
    recoil = list(config.get("recoil", {}).keys())
    followers = list(config.get("followers", []))
    items = ([(o, "born") for o in born] + [(o, "recoil") for o in recoil]
             + [(o, "follower") for o in followers])
    items = [(o, r) for (o, r) in items if o in events]
    n = len(items)
    labels = labels or {}
    log_obs = set(log_obs or [o for o, r in items if r == "recoil"] + ["phistar"])
    fo = fo or {}

    wpr = np.asarray(events[weight_key], float)
    wpo = np.asarray(result.weights, float)
    band = getattr(result, "band", None)

    fig, ax = plt.subplots(2, n, figsize=(4.7 * n, 6.6), squeeze=False,
                           gridspec_kw={"height_ratios": [2.1, 1.25],
                                        "hspace": 0.06, "wspace": 0.26})
    role_tag = {"recoil": "recoil - composite window",
                "born": "Born - full range",
                "follower": "follower - never constrained"}

    for j, (obs, role) in enumerate(items):
        x = np.asarray(events[obs], float)
        e = _edges(x, obs, config, data)
        ctr = np.sqrt(e[:-1] * e[1:]) if (e > 0).all() else 0.5 * (e[:-1] + e[1:])
        hpr = _dens(x, wpr, e)
        hpo = _dens(x, wpo, e)
        sr = _statrel(x, wpo, e)
        if band:
            hb = np.array([_dens(x, np.asarray(bw, float), e) for bw in band.values()])
            lo_b = np.minimum(hb.min(0), hpo) * (1 - sr)
            hi_b = np.maximum(hb.max(0), hpo) * (1 + sr)
            band_lbl = r"MaxEnt $\pm$(scale$\oplus$stat)"
        else:
            lo_b, hi_b = hpo * (1 - sr), hpo * (1 + sr)
            band_lbl = r"MaxEnt $\pm$stat"

        a, r = ax[0, j], ax[1, j]
        # ---- reference: PS+LO prior (always)
        a.stairs(hpr, e, ls="--", color=C_PRIOR, lw=1.4, label="PS+LO prior")
        # ---- reference: fixed order (always when provided)
        fo_ctr = fo_val = None
        if obs in fo:
            fe, fo_ctr, fo_val = _fo_density(fo[obs], e, x, wpo)
            a.stairs(fo_val, fe, ls=":", color=C_FO, lw=1.7, label="FO (NNLOJET)")
        # ---- the upgrade result + band
        a.stairs(hi_b, e, baseline=lo_b, fill=True, color=C_POST, alpha=0.25)
        a.stairs(hpo, e, color=C_POST, lw=2.1, label=band_lbl)
        # ---- matched generators (optional)
        gden = {}
        if generators:
            for gi, (gname, g) in enumerate(generators.items()):
                if not (isinstance(g, dict) and (obs in g or "x" in g)):
                    continue
                gx = np.asarray(g.get(obs, g.get("x")), float)
                gw = np.asarray(g.get("weight", g.get("w")), float)
                if gx is None or gw is None or len(gx) != len(gw):
                    continue
                gd = _dens(gx, gw, e); gden[gname] = gd
                a.stairs(gd, e, color=C_GEN[gi % 3], lw=1.5, label=gname)
        # ---- data (when available)
        has_data = bool(data and obs in data)
        if has_data:
            D = data[obs]
            val, err = np.asarray(D["val"], float), np.asarray(D["err"], float)
            m = val > 0
            a.errorbar(ctr[m], val[m], yerr=err[m], fmt="o", color="k", ms=3,
                       lw=1, label=data_label, zorder=6)
        # ratio reference in priority order: data -> FO -> prior (never ratio-to-self)
        if has_data:
            ref = np.maximum(val, 1e-30); ref_name = "data"
        elif fo_val is not None:
            ref = np.maximum(np.interp(ctr, fo_ctr, fo_val, left=np.nan, right=np.nan), 1e-30)
            ref_name = "FO"
        else:
            ref = np.maximum(hpr, 1e-30); ref_name = "prior"

        if role == "recoil":
            xm = float(result.x_match)
            xhi = config["recoil"][obs].get("range", [None, e[-1]])[1] or e[-1]
            for p in (a, r):
                p.axvspan(xm, xhi, color=C_WIN, alpha=0.16)

        if obs in log_obs:
            a.set_xscale("log"); r.set_xscale("log")
        a.set_yscale("log"); a.grid(alpha=0.22, which="both")
        a.tick_params(labelbottom=False)
        a.text(0.96, 0.94, role_tag[role], transform=a.transAxes, ha="right",
               va="top", fontsize=9, style="italic", color="0.4")
        a.set_title(labels.get(obs, obs))

        # ---- ratio sub-panel (to data if given, else to MaxEnt central)
        r.axhline(1, color="k", lw=0.8)
        r.stairs(hpr / ref, e, ls="--", color=C_PRIOR, lw=1.3)
        if fo_val is not None:
            fo_i = np.interp(ctr, fo_ctr, fo_val, left=np.nan, right=np.nan)
            r.stairs(fo_i / ref, e, ls=":", color=C_FO, lw=1.5)
        r.stairs(hi_b / ref, e, baseline=lo_b / ref, fill=True, color=C_POST, alpha=0.25)
        r.stairs(hpo / ref, e, color=C_POST, lw=2.0)
        for gi, (gname, gd) in enumerate(gden.items()):
            r.stairs(gd / ref, e, color=C_GEN[gi % 3], lw=1.3)
        r.set_ylim(0.6, 1.4); r.grid(alpha=0.22)
        r.set_xlabel(labels.get(obs, obs))
        good = np.isfinite(ref) & (ref > 0)
        med = 100 * np.median(np.abs(hpo[good] / ref[good] - 1)) if good.any() else np.nan
        r.text(0.03, 0.06, f"MaxEnt vs {ref_name}: med|r-1|={med:.1f}%",
               transform=r.transAxes, fontsize=8, color=C_POST)
        if j == 0:
            a.set_ylabel(r"$(1/\sigma)\,d\sigma/dX$")
            r.set_ylabel("ratio to ref")
            a.legend(fontsize=8, loc="lower left")

    negwt = 100 * np.mean(wpo <= 0)
    head = (f"effN={100*result.effN:.0f}%   neg-wt={negwt:.0f}%   "
            f"closure={result.closure:.1e}   x_match={result.x_match:.3g} GeV   "
            f"moments {result.chosen_moments}")
    fig.suptitle((title + "\n" if title else "") + head, fontsize=11, y=0.99)
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outfile
