#!/usr/bin/env python3
r"""Shared uncertainty-band helpers for the data-comparison figures.

One convention everywhere:
  scale -> translucent band (per-bin envelope over the 7-point variants)
  stat  -> error bars (per-bin MC error of the sample; for MaxEnt combined
           with the moment-bootstrap spread), horizontally staggered per
           series so overlapping bars stay legible.
All densities are normalized shapes on the histogram edges, so every band is
a SHAPE uncertainty, consistent with every central curve drawn.
"""
import numpy as np


def norm_dens(x, w, e):
    """(1/sigma) dsigma/dX with sigma = TOTAL weight sum (ATLAS convention for
    normalized distributions), NOT the in-range integral -- the data figures
    normalize to the fiducial cross section, overflow included."""
    w = np.asarray(w, float)
    h, _ = np.histogram(np.asarray(x, float), e, weights=w)
    return h / np.diff(e) / w.sum()


def scale_envelope(x, w_scale, e, central=None):
    """Per-bin (lo, hi) of the normalized shape over scale-variant weights."""
    hs = np.array([norm_dens(x, w_scale[:, k], e)
                   for k in range(w_scale.shape[1])])
    lo, hi = hs.min(0), hs.max(0)
    if central is not None:
        lo, hi = np.minimum(lo, central), np.maximum(hi, central)
    return lo, hi


def mc_err(x, w, e):
    """Per-bin MC error of the normalized shape: sqrt(sum w^2), same TOTAL
    normalization as norm_dens."""
    w = np.asarray(w, float)
    s2, _ = np.histogram(np.asarray(x, float), e, weights=w * w)
    return np.sqrt(s2) / np.diff(e) / abs(w.sum())


def stagger(e, k, n, frac=0.30):
    """Bin centers shifted for series k of n, keeping bars inside the bin."""
    lo, hi = e[:-1], e[1:]
    t = 0.5 + frac * ((k + 0.5) / n - 0.5) * 2
    return lo + t * (hi - lo)


def draw_series_unc(ax, x, w, w_scale, e, ref, color, k, n, boot=None,
                    alpha=0.14, bars=True):
    """Scale band + staggered stat bars for one sample, in RATIO to ref."""
    h = norm_dens(x, w, e)
    ok = np.isfinite(ref) & (ref > 0)
    r = np.where(ok, h / np.where(ok, ref, 1), np.nan)
    if w_scale is not None:
        lo, hi = scale_envelope(x, w_scale, e, central=h)
        ax.fill_between(0.5 * (e[:-1] + e[1:]),
                        np.where(ok, lo / np.where(ok, ref, 1), np.nan),
                        np.where(ok, hi / np.where(ok, ref, 1), np.nan),
                        step="mid", color=color, alpha=alpha, lw=0)
    if bars:
        err = mc_err(x, w, e)
        if boot is not None:
            err = np.hypot(err, boot)
        ax.errorbar(stagger(e, k, n), r, yerr=np.where(ok, err / np.where(ok, ref, 1), np.nan),
                    fmt="none", ecolor=color, elinewidth=1.1, capsize=1.8, alpha=0.9)
