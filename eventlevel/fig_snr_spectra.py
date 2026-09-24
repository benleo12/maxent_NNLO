#!/usr/bin/env python3
r"""JHEP figure: resolution AND saturation, side by side.

Left: SNR_n = |mu_n^target - mu_n^prior| / sigma_FO(mu_n) for the three
constrained Drell-Yan observables, straight from the solver's own report
(res.moment_snr), with sigma_FO the seed scatter combined with the scale
envelope.  Every booked moment clears the threshold, so resolution does NOT
set the truncation -- which invites the question "then why stop at 12 and 20?"

Both towers are continued past their booked depth (Born 6, recoil 12) to a
common n=14 with OPEN markers, using the sub-bin histogram estimator of
audit_moments2.hist_moment; it reproduces every booked accumulator SNR point
(worst deviation quoted at run time), so the continuation is a measurement,
not an extrapolation.

Right: the answer.  Re-solving at every tower depth, the delivered spectra
stop moving: the mll median deviation from NNLO and the above-seam pT_ll
median deviation from the NLO(Zj) curve both plateau well before the booked
depths, with effN flat.  Deeper towers are pure cost (more accumulators to
book in NNLOJET), not more physics.  Both scans run to depth 12; the Born
points beyond 6, where NNLOJET booked no accumulator, impose the same
histogram-derived targets and are drawn with open markers.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubstyle import use_pub_style, C, rebin_density
use_pub_style(base=17)
from maxent_upgrade import upgrade
from nnlojet_moments import (fo_moments_smooth_from_nnlojet, common_seeds,
                             fo_curve, _load)

BASE = os.path.join(os.environ.get("NNLOJET_ROOT",
        os.path.expanduser("~/nnlojet-v1.0.2")), "dy_profile_log30_hi")
CH6 = ["LO", "R", "V", "RR", "RV", "VV"]
XM, XHI, SOFT, XMAP = 30.0, 500.0, 30.0, 2500.0


def cfg():
    return dict(born={"mll": {"range": (66., 116.), "map": "bw"},
                      "y_abs": {"range": (0., 2.4), "map": "lin"}},
                recoil={"pT_ll": {"range": (SOFT, XMAP), "map": "log", "soft_lo": SOFT,
                                  "profile": {"a": XM, "b": 2 * XM, "c": XHI}}})


def main():
    P = dict(np.load(os.path.join(HERE, "dy_prior_atlas_v3.npz")))
    n = len(P["w"]); idx = np.random.default_rng(0).choice(n, min(1_000_000, n), replace=False)
    ev = dict(mll=P["mll"][idx].astype(float), y_abs=np.abs(P["y_ll"][idx]).astype(float),
              pT_ll=P["pT_ll"][idx].astype(float), weight=P["w"][idx].astype(float))
    seeds = common_seeds(BASE, "DY_MOMENTS", CH6)

    def moments(nb, nr):
        return fo_moments_smooth_from_nnlojet(BASE, "DY_MOMENTS", CH6, seeds,
                                              born_tags={"mll": "mll", "y_abs": "absyz"},
                                              n_born=nb, n_recoil=nr,
                                              x_match=XM, x_hi=XMAP, soft_lo=SOFT)

    res = upgrade(ev, moments(12, 20), cfg())
    print(f"solve: effN {100*res.effN:.1f}%  closure {res.closure:.2e}")

    # ---- FO reference curves for the saturation scan ----------------------
    e_m = np.linspace(66., 116., 26); bw_m = np.diff(e_m)
    flo, fhi, fd, _ = fo_curve(BASE, "DY_MOMENTS", CH6,
                               common_seeds(BASE, "DY_MOMENTS", CH6, tag="mll_fine"),
                               "mll_fine")
    g = (fhi > flo) & np.isfinite(fd)
    fo_m = rebin_density(flo[g], fhi[g], fd[g], e_m)
    fo_m = fo_m / np.nansum(fo_m * bw_m)

    e_p = np.geomspace(XM, XHI, 16); bw_p = np.diff(e_p)
    tot = None
    for s_ in seeds:
        for ch in ("R", "RR", "RV"):
            r0 = _load(os.path.join(BASE, f"ch_{ch}",
                                    f"Z.DY_MOMENTS.{ch}.ptz_winfine.s{s_}.dat"))
            if r0 is None:
                continue
            plo, _, phi_, v, _ = r0
            tot = v[:, 0].copy() if tot is None else tot + v[:, 0]
    gp = (tot > 0) & (phi_ > plo)
    fo_p = rebin_density(plo[gp], phi_[gp], tot[gp], e_p)
    fo_p = fo_p / np.nansum(fo_p * bw_p)          # shape-only within the window

    def med_mll(w):
        h, _ = np.histogram(ev["mll"], e_m, weights=w)
        h = h / bw_m; h = h / (h * bw_m).sum()
        m = np.isfinite(fo_m) & (fo_m > 0)
        return 100 * np.median(np.abs(h[m] / fo_m[m] - 1))

    # |y_ll| reference (24 bins, as fig_dy_spectra) for the Born-depth scan
    e_y = np.linspace(0., 2.4, 25); bw_y = np.diff(e_y)
    ylo, yhi, yd, _ = fo_curve(BASE, "DY_MOMENTS", CH6,
                               common_seeds(BASE, "DY_MOMENTS", CH6, tag="absyz_fine"),
                               "absyz_fine")
    gy = (yhi > ylo) & np.isfinite(yd)
    fo_y = rebin_density(ylo[gy], yhi[gy], yd[gy], e_y)
    fo_y = fo_y / np.nansum(fo_y * bw_y)

    def med_yll(w):
        h, _ = np.histogram(ev["y_abs"], e_y, weights=w)
        h = h / bw_y; h = h / (h * bw_y).sum()
        m = np.isfinite(fo_y) & (fo_y > 0)
        return 100 * np.median(np.abs(h[m] / fo_y[m] - 1))

    def med_ptw(w):
        inw = (ev["pT_ll"] >= XM) & (ev["pT_ll"] < XHI)
        h, _ = np.histogram(ev["pT_ll"][inw], e_p, weights=w[inw])
        h = h / bw_p; h = h / (h * bw_p).sum()
        m = np.isfinite(fo_p) & (fo_p > 0) & (h > 0)
        return 100 * np.median(np.abs(h[m] / fo_p[m] - 1))

    # ---- EXTENDED towers: where does each observable actually cross SNR=1?
    # The booked accumulators stop at 12 (Born) / 20 (recoil), so higher orders
    # are estimated from the per-seed FINE HISTOGRAMS -- targets and
    # (stat + scale) errors alike -- by the SUB-BIN integration of
    # audit_moments2.hist_moment: every bin's density is reconstructed as a
    # minmod-limited linear profile in x (exact bin integral) and w(x) T_n(u(x))
    # is averaged over NSUB sub-points.  Evaluating T_n and the profile at BIN
    # CENTERS instead is what previously made the recoil estimate fail its own
    # validation; with the sub-bin integral -- and with the prior side taken
    # over the SAME hard window the solver uses in `_impose` -- it reproduces
    # every booked accumulator SNR point (see the printed check below).
    MZ, GZ = 91.1876, 2.4952
    NSUB = 32

    def u_bw(x):
        t = np.arctan((np.clip(x, 66., 116.) ** 2 - MZ ** 2) / (MZ * GZ))
        ta = np.arctan((66. ** 2 - MZ ** 2) / (MZ * GZ))
        tb = np.arctan((116. ** 2 - MZ ** 2) / (MZ * GZ))
        return np.clip(2 * (t - ta) / (tb - ta) - 1, -1, 1)

    def u_lin(x):
        # the compiled eval_chebT_absyz clips out-of-range events to the edge;
        # the absyz_fine histogram runs to |y|=6, so the estimator must clip too
        return np.clip(2 * x / 2.4 - 1, -1, 1)

    def u_logw(x):
        # the compiled map of the recoil tower: log [SOFT, XMAP] (the window
        # [XM, XHI) is separate and enters through wprof / the hard mask)
        return np.clip(2 * (np.log(np.clip(x, SOFT, XMAP)) - np.log(SOFT))
                       / (np.log(XMAP) - np.log(SOFT)) - 1, -1, 1)

    def wprof(x):
        t = np.clip(np.log(np.maximum(x, 1e-9) / XM) / np.log(2.0), 0, 1)
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    def cheb_mat(u, nmax):
        T = [np.ones_like(u), u]
        for _ in range(2, nmax + 1):
            T.append(2 * u * T[-1] - T[-2])
        return np.array(T[1:nmax + 1])

    def hist_tower(tag, channels, umap, nmax, wfn=None, rebin=1):
        """<w T_n>/<w> and its (stat (+) scale) error from a fine histogram."""
        sds = common_seeds(BASE, "DY_MOMENTS", CH6, tag=tag)
        frac = (np.arange(NSUB) + 0.5) / NSUB
        P = N = None
        rows = []
        for sd in sds:
            tot = None
            for ch in channels:
                r0 = _load(os.path.join(BASE, f"ch_{ch}",
                                        f"Z.DY_MOMENTS.{ch}.{tag}.s{sd}.dat"))
                if r0 is None:
                    continue
                lo_, _, hi_, v, _ = r0
                tot = v.copy() if tot is None else tot + v
            if tot is None:
                continue
            g = hi_ > lo_
            lo, hi, d = lo_[g], hi_[g], tot[g]
            if rebin > 1:                       # discretisation self-check only
                k = (len(lo) // rebin) * rebin
                w0 = (hi - lo)[:k]
                a_ = (d[:k] * w0[:, None]).reshape(-1, rebin, d.shape[1]).sum(1)
                lo, hi = lo[:k].reshape(-1, rebin)[:, 0], hi[:k].reshape(-1, rebin)[:, -1]
                d = a_ / (hi - lo)[:, None]
            bw = hi - lo; mid = 0.5 * (lo + hi)
            xs = lo[:, None] + bw[:, None] * frac[None, :]          # (nbin, NSUB)
            T = cheb_mat(umap(xs.ravel()), nmax).reshape(nmax, len(bw), NSUB)
            wm = wfn(xs) if wfn is not None else np.ones_like(xs)
            # minmod-limited linear density inside each bin (integral preserved)
            dl = np.zeros_like(d); dr = np.zeros_like(d)
            dl[1:] = (d[1:] - d[:-1]) / np.diff(mid)[:, None]
            dr[:-1] = dl[1:]
            sl = np.where(dl * dr > 0,
                          np.sign(dl) * np.minimum(np.abs(dl), np.abs(dr)), 0.0)
            rho = d[:, None, :] + sl[:, None, :] * (xs - mid[:, None])[:, :, None]
            base = rho * wm[:, :, None] * bw[:, None, None]
            num = np.einsum("nbj,bjs->ns", T, base) / NSUB
            den = base.sum((0, 1)) / NSUB
            P = num if P is None else P + num
            N = den if N is None else N + den
            rows.append(num[:, 0] / den[0])
        A = np.asarray(rows)
        pooled = P / N[None, :]                      # pooled MC, per scale
        stat = A.std(0, ddof=1) / np.sqrt(len(A))
        return pooled[:, 0], np.hypot(stat, 0.5 * (pooled.max(1) - pooled.min(1)))

    def prior_tower(key, nmax):
        """Prior-side moment on EXACTLY the support `_impose` uses: the full
        fiducial sample for a Born observable, the hard window [XM, XHI) for
        the recoil (the solver's `mu_prior_win`)."""
        x = np.asarray(ev[key], float); w = ev["weight"]
        if key == "pT_ll":
            m = (x >= XM) & (x < XHI)
            x, w = x[m], w[m]
            umap = u_logw
        else:
            umap = u_bw if key == "mll" else u_lin
        T = cheb_mat(umap(x), nmax)
        return (T * w).sum(1) / w.sum()

    NEXT = 20                       # common extended depth for all three towers
    EXT = {"mll":   ("mll_fine",    CH6,               u_bw,   None),
           "y_abs": ("absyz_fine",  CH6,               u_lin,  None),
           "pT_ll": ("ptz_winfine", ("R", "RR", "RV"), u_logw, wprof)}
    ext_snr, ext_cross, ext_mu, ext_sig = {}, {}, {}, {}
    for key, (tag, chs, umap, wfn) in EXT.items():
        tgt, sig = hist_tower(tag, chs, umap, NEXT, wfn)
        tgt2, _ = hist_tower(tag, chs, umap, NEXT, wfn, rebin=2)
        mup = prior_tower(key, NEXT)
        snr_e = np.abs(tgt - mup) / np.maximum(sig, 1e-30)
        ext_snr[key] = snr_e; ext_mu[key] = tgt; ext_sig[key] = sig
        below = np.where(snr_e < 1.0)[0]
        ext_cross[key] = (below[0] + 1) if len(below) else None
        booked = np.asarray(res.moment_snr[key], float)
        nb_ = len(booked)
        print(f"  {key}: hist-SNR " + " ".join(f"{s_:.3g}" for s_ in snr_e))
        print(f"         booked-order check (hist / solver accumulator SNR): "
              + " ".join(f"{snr_e[i]:.1f}/{booked[i]:.1f}" for i in range(nb_)))
        print(f"         worst booked-order SNR deviation: "
              f"{100*np.max(np.abs(snr_e[:nb_] / booked - 1)):.2f}%")
        print(f"         discretisation self-check, max |mu(bins) - mu(bins/2)| "
              f"over n<={NEXT}: {np.max(np.abs(tgt - tgt2)):.2e} "
              f"(vs sigma_FO {np.median(sig):.2e})")
        print(f"         crossing: n = {ext_cross[key]}")

    # ---- depth scans: Born tower at fixed recoil, recoil tower at fixed Born
    # Beyond the booked Born depth of 12 the imposed targets come from the same
    # validated histogram estimator, so both scans reach the same depth.
    NSCAN_B, NSCAN_R = 20, 20        # Born scan continues past the booked 12 with histogram-derived targets

    def moments_ext(nb, nr):
        m = moments(min(nb, 12), nr)
        for o in ("mll", "y_abs"):
            v = list(m["born"][o]["values"]); e = list(m["born"][o]["errors"])
            v += list(ext_mu[o][12:nb]); e += list(ext_sig[o][12:nb])
            m["born"][o].update(values=v, errors=e)
        return m

    born_scan, recoil_scan = [], []
    for nb in range(1, NSCAN_B + 1):
        try:
            r_ = upgrade(ev, moments_ext(nb, 20), cfg())
        except Exception as exc:                       # pragma: no cover
            print(f"  !! born n<={nb} did NOT converge ({exc}); scan stops here")
            break
        born_scan.append((nb, med_mll(r_.weights), 100 * r_.effN, med_yll(r_.weights)))
        got = r_.chosen_moments["mll"]
        print(f"  born n<={nb}:  mll dev {born_scan[-1][1]:5.2f}%   |y| dev {born_scan[-1][3]:5.2f}%   "
              f"effN {100*r_.effN:.1f}%   imposed {got}"
              + ("  !! SNR selection truncated the tower" if got != nb else "")
              + ("  (orders >12 histogram-derived)" if nb > 12 else ""))
    for nr in range(1, NSCAN_R + 1):
        try:
            r_ = upgrade(ev, moments(12, nr), cfg())
        except Exception as exc:                       # pragma: no cover
            print(f"  !! recoil n<={nr} did NOT converge ({exc}); scan stops here")
            break
        recoil_scan.append((nr, med_ptw(r_.weights), 100 * r_.effN))
        got = r_.chosen_moments["pT_ll"]
        print(f"  recoil n<={nr}: pT_ll dev {recoil_scan[-1][1]:5.2f}%   "
              f"effN {100*r_.effN:.1f}%   imposed {got}"
              + ("  !! SNR selection truncated the tower" if got != nr else ""))

    # ---- figure -----------------------------------------------------------
    fig, (ax, sx) = plt.subplots(1, 2, figsize=(13.4, 5.2),
                                 gridspec_kw={"wspace": 0.26})
    series = [(r"$m_{\ell\ell}$ (BW map, NNLO)", "mll", C["maxent"], "o"),
              (r"$|y_{\ell\ell}|$ (linear map, NNLO)", "y_abs", C["minnlo"], "s"),
              (r"$p_T^{\ell\ell}$ (windowed, NLO $Z$+jet)", "pT_ll", C["powheg"], "D")]
    for lbl, key, col, mk in series:
        snr = np.asarray(res.moment_snr[key], float)
        nb = len(snr)
        nn = np.arange(1, nb + 1)
        ax.plot(nn, snr, mk + "-", color=col, ms=7, lw=1.8, label=lbl)
        # histogram-derived continuation past the booked depth, open markers
        se = ext_snr[key]
        ne = np.arange(nb + 1, len(se) + 1)
        if len(ne):
            ax.plot(np.concatenate([[nb], ne]),
                    np.concatenate([[snr[-1]], se[nb:]]),
                    mk + "--", color=col, ms=7, lw=1.2, mfc="none", alpha=0.8)
        xc = ext_cross[key]
        if xc is not None:
            ax.annotate(rf"$n={xc}$", (xc, se[xc - 1]),
                        textcoords="offset points", xytext=(6, -14),
                        fontsize=12, color=col)
        print(f"  {key}: SNR " + " ".join(f"{s:.3g}" for s in snr))
    ax.axhline(1.0, color="k", lw=1.4, ls="--")
    ax.text(0.75, 1.13, r"$\mathrm{SNR}=1$", va="bottom", fontsize=13)
    ax.set_yscale("log")
    ax.set_xlabel(r"moment order $n$")
    ax.set_ylabel(r"$\mathrm{SNR}_n=|\mu_n^{\rm target}-\mu_n^{\rm prior}|/\sigma^{\rm FO}(\mu_n)$")
    ax.set_xticks(range(1, 21)); ax.set_xlim(0.5, 20.9)
    # headroom so the deepest open markers clear the legend and the footnote
    ax.set_ylim(0.08, 6e3)
    ax.legend(loc="upper right", fontsize=12, labelspacing=0.3)

    ax.text(0.02, 0.03, r"open markers: beyond the booked depth",
            transform=ax.transAxes, fontsize=11, color="0.35")

    bs = np.array(born_scan); rs = np.array(recoil_scan)
    # solid where every imposed moment is a booked accumulator, open beyond it
    ys = bs[:, [0, 3]]
    for arr, col, mk, nbook, lbl in (
            (bs[:, :2], C["maxent"], "o", 12,
             r"$m_{\ell\ell}$ vs NNLO, Born tower depth"),
            (ys, C["minnlo"], "s", 12,
             r"$|y_{\ell\ell}|$ vs NNLO, Born tower depth"),
            (rs, C["powheg"], "D", 20,
             r"$p_T^{\ell\ell}$ (window) vs NLO $Zj$, recoil depth")):
        ms = 8 if mk == "o" else 7
        sx.plot(arr[:, 0], arr[:, 1], mk + "-", color=col, ms=ms, lw=2.0, label=lbl)
        beyond = arr[:, 0] > nbook
        if beyond.any():
            sx.plot(arr[beyond, 0], arr[beyond, 1], mk, color=col, ms=ms,
                    mfc="white", mew=1.6)
    s2 = sx.twinx()
    s2.plot(bs[:, 0], bs[:, 2], "o:", color=C["maxent"], ms=4, lw=1.2, alpha=0.55)
    s2.plot(rs[:, 0], rs[:, 2], "D:", color=C["powheg"], ms=4, lw=1.2, alpha=0.55)
    s2.set_ylabel(r"$N_\mathrm{eff}/N$ [\%] (dotted)", fontsize=13)
    s2.set_ylim(0, 100)
    sx.set_xlabel(r"imposed tower depth $N_X$")
    sx.set_ylabel(r"median $|$ratio$-1|$ [\%]")
    sx.set_xticks(range(1, 21)); sx.set_xlim(0.5, 20.5)
    sx.set_ylim(0, None)
    # centre-right: clear of the depth-1 point and of the flat effN dotted lines
    sx.legend(loc="center right", fontsize=12, labelspacing=0.3)
    out = os.path.join(HERE, "fig_snr_spectra.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png")); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
