#!/usr/bin/env python3
r"""maxent_upgrade -- upgrade a PS+LO event sample to PS+N^kLO by MaxEnt moment reweighting.

ONE public entry point::

    from maxent_upgrade import upgrade
    result = upgrade(events, fo_low, fo_high, config)

`result.weights` are new, strictly-positive per-event weights that carry the prior
sample from order N to order N+k in the *constrained moments and in every observable
kinematically determined by them*, while preserving the prior parton-shower accuracy
everywhere the fixed-order (FO) input does not resolve a change.  No negative weights
are ever produced (the reweighting is an exponential tilt of the prior, q = p * e^{lambda . phi}).

This module is a THIN, documented wrapper around a frozen, already-correct physics
engine (`maxent_match.py`, functions `build_and_solve` / `band_solve` / the convex-dual
Newton solver `_newton_maxent`, plus `dy_method.cheb`/`umap`/`maxent`).  It does NOT
re-implement any physics.  It only:

  * translates a friendly per-observable `config` into the engine's internal `cfg`;
  * assembles the user's per-observable FO histograms into the engine's multi-part
    container and wires up the per-observable histogram dispatch;
  * chooses the number of constrained Chebyshev moments PER observable from the
    moment signal-to-noise spectrum (`moment_snr`), the systematic replacement for
    hand-picked moment counts;
  * packages the outputs (weights, effN, closure, x_match, scale/rate band, report)
    into a single `UpgradeResult`.

See `README.md` for the physics contract and the full list of knobs.

Public API
----------
    upgrade(events, fo_low, fo_high, config) -> UpgradeResult
    moment_snr(fo_low, fo_high, events, obs, a, b, mp, Nmax, ...) -> np.ndarray
    resolved_order(snr, threshold=1.0) -> int
    FOHist(bin_low, bin_high, value, error, scales=None, scale_errors=None)
    fo_from_dat(paths, nscales=7) -> FOHist          # NNLOJET-style .dat reader
    UpgradeResult                                     # dataclass returned by upgrade()
    DEFAULTS                                          # dict of default knobs
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import numpy as np

# ---- the frozen physics engine (verbatim copy shipped inside this package) ----
from . import maxent_match as _mm
from .maxent_match import (
    build_and_solve as _build_and_solve,
    band_solve as _band_solve,
    kconv_xmatch as _kconv_xmatch,
    ps_agreement_xmatch as _ps_agreement_xmatch,
    fo_from_dat as _engine_fo_from_dat,
    DIALS as _DIALS,
)
from .dy_method import cheb as _cheb, umap as _umap

__all__ = [
    "upgrade", "moment_snr", "resolved_order",
    "FOHist", "fo_from_dat", "UpgradeResult", "DEFAULTS",
]

# ---------------------------------------------------------------------------
# Default knobs (documented in README.md).  A `config` may override any of them.
# ---------------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = dict(
    weight_key="weight",     # key in `events` holding the positive prior weights
    # --- moment-SNR selection (the systematic replacement for fixed counts) ---
    moment_selection=True,   # choose #moments per obs from SNR>threshold
    snr_threshold=1.0,       # a moment is "resolved by the FO calc" when SNR_n > this
    snr_max_order=dict(born=10, recoil=12),  # ceiling of the SNR scan per role
    snr_n_stat=200,          # MC resamples for the FO statistical component of sigma
    snr_seed=0,
    # --- engine dials (see maxent_match.DIALS); override selectively ----------
    L2=_DIALS["L2"],         # ridge on the dual (closure/roughness trade-off)
    RELMAX=_DIALS["RELMAX"], # diff-K / x_match validity: drop FO bins with rel err > this
    DELTA=_DIALS["DELTA"],   # plateau tolerance for the x_match convergence rule
    WSIG=_DIALS["WSIG"],     # recoil-window admission by significance (val/err > WSIG)
    CLIP=_DIALS["CLIP"],     # clamp on the differential K-factor
    MINSUP=_DIALS["MINSUP"], # min prior events per recoil bin to trust it
    RSUP=_DIALS["RSUP"],     # max target/prior probability ratio per recoil bin
    # --- band -----------------------------------------------------------------
    band=True,               # also solve every FO scale variation + rate scheme
    # --- fixed counts (used only when moment_selection is False) --------------
    born_N=None,             # {obs: n}; None -> engine default N_FO for every Born obs
    recoil_N=None,           # int; None -> engine default N_W
)

# engine dials that a config may override (name -> DIALS key)
_DIAL_KEYS = ("L2", "RELMAX", "DELTA", "WSIG", "CLIP", "MINSUP", "RSUP")


# ---------------------------------------------------------------------------
# FO histogram container
# ---------------------------------------------------------------------------
@dataclass
class FOHist:
    """A fixed-order differential distribution the USER provides for one observable.

    Parameters
    ----------
    bin_low, bin_high : 1D arrays
        Bin edges (lower/upper) of the FO histogram, length = n_bins.
    value : 1D array
        Central-scale differential cross section per bin (any units; only shapes and
        ratios of the two orders are used).  length = n_bins.
    error : 1D array
        Monte-Carlo statistical error on `value`, per bin.  length = n_bins.
    scales : (n_scale, n_bin) array, optional
        Scale-variation cross sections.  Row 0 MUST be the central scale (== `value`).
        Rows 1.. are the members of the scale envelope (e.g. the 7-point set).
        If omitted, no scale band is produced and the uncertainty comes from `error`
        alone.
    scale_errors : (n_scale, n_bin) array, optional
        MC error for each scale row.  If omitted, `error` is reused for every scale.
    """
    bin_low: np.ndarray
    bin_high: np.ndarray
    value: np.ndarray
    error: np.ndarray
    scales: Optional[np.ndarray] = None
    scale_errors: Optional[np.ndarray] = None

    def __post_init__(self):
        self.bin_low = np.asarray(self.bin_low, float)
        self.bin_high = np.asarray(self.bin_high, float)
        self.value = np.asarray(self.value, float)
        self.error = np.asarray(self.error, float)
        n = len(self.bin_low)
        for nm in ("bin_high", "value", "error"):
            if len(getattr(self, nm)) != n:
                raise ValueError(f"FOHist: '{nm}' length {len(getattr(self,nm))} != n_bins {n}")
        if self.scales is not None:
            self.scales = np.atleast_2d(np.asarray(self.scales, float))
            if self.scales.shape[1] != n:
                raise ValueError("FOHist: scales second axis must be n_bins")
        if self.scale_errors is not None:
            self.scale_errors = np.atleast_2d(np.asarray(self.scale_errors, float))


def _as_dat(fh) -> dict:
    """Convert a FOHist (or an already-engine 'dat' dict) into the engine dat dict:
       {lo, hi, vals:[nsc arrays], errs:[nsc arrays], kind:'dat'} sorted by lo."""
    if isinstance(fh, dict) and fh.get("kind") == "dat":
        return fh
    if not isinstance(fh, FOHist):
        raise TypeError("FO input must be a FOHist or an engine dat dict")
    lo, hi = fh.bin_low, fh.bin_high
    if fh.scales is not None:
        vals = [fh.scales[k] for k in range(fh.scales.shape[0])]
        if fh.scale_errors is not None and fh.scale_errors.shape[0] == len(vals):
            errs = [fh.scale_errors[k] for k in range(len(vals))]
        else:
            errs = [fh.error for _ in vals]
    else:
        vals = [fh.value]
        errs = [fh.error]
    o = np.argsort(lo)
    return dict(lo=lo[o], hi=hi[o], vals=[v[o] for v in vals],
                errs=[e[o] for e in errs], kind="dat")


def _edges(dat: dict) -> np.ndarray:
    return np.concatenate([dat["lo"][:1], dat["hi"]])


# ---------------------------------------------------------------------------
# NNLOJET-style .dat reader -> FOHist  (lo ce hi val err val err ... nscales)
# ---------------------------------------------------------------------------
def fo_from_dat(paths, nscales: int = 7) -> FOHist:
    """Read one-or-more NNLOJET '.dat' density files into a single FOHist.

    Multiple paths are concatenated and sorted by lower edge (e.g. ptz_fine + ptz_mid
    + ptz_high tile one recoil spectrum).  Column layout per row:
        lo  center  hi  val_1 err_1  val_2 err_2  ...  val_{nscales} err_{nscales}
    where scale 1 is the central scale.  Returns a FOHist whose `scales` are all
    `nscales` variations (row 0 = central) so `upgrade` can build the scale band.
    """
    if isinstance(paths, str):
        paths = [paths]
    d = _engine_fo_from_dat(list(paths), nscales=nscales)  # engine dat dict
    scales = np.array(d["vals"])          # (nscales, nbin)
    scale_errors = np.array(d["errs"])    # (nscales, nbin)
    return FOHist(bin_low=d["lo"], bin_high=d["hi"], value=scales[0], error=scale_errors[0],
                  scales=scales, scale_errors=scale_errors)


# ---------------------------------------------------------------------------
# Per-moment signal-to-noise spectrum (folds in scratchpad/moment_snr.py)
# ---------------------------------------------------------------------------
def moment_snr(fo_low, fo_high, events, obs, a, b, mp, Nmax,
               weight_key="weight", window=None, n_stat=200, seed=0, n_events_max=300000):
    r"""Per-moment signal-to-noise spectrum for one observable.

        SNR_n = | mu_n(target) - mu_n(prior) | / sigma_FO(mu_n)

    where mu_n is the n-th Chebyshev moment (order n = 1..Nmax) on the variable map
    `mp` ('lin'/'log') over [a, b].  The target moment is the event-level average of
    T_n(u) reweighted by the binned differential K-factor K(x) = dens_{N+k}(x)/dens_N(x)
    (for a windowed recoil, `window=(x_match, x_hi)` restricts to the FO region and
    remaps).  sigma_FO(mu_n) is propagated by Monte Carlo from BOTH FO uncertainty
    sources: the scale envelope (half-spread of mu_n over the scale rows) and the FO
    statistical errors (resample every bin of each order within its error, recompute
    the moment; `n_stat` resamples), added in quadrature.

    A moment is "genuinely resolved by the FO calculation" when SNR_n > 1: the change
    the FO calc predicts in that moment exceeds the FO's own uncertainty on it.

    Returns
    -------
    snr : np.ndarray, shape (Nmax,)
        SNR for moment orders n = 1 .. Nmax.
    """
    rng = np.random.default_rng(seed)
    hi = _as_dat(fo_high)
    lo = _as_dat(fo_low)
    xlo, xhi = hi["lo"], hi["hi"]
    vN = np.array(hi["vals"]); eN = np.array(hi["errs"])   # (nsc, nbin)
    vL = np.array(lo["vals"]); eL = np.array(lo["errs"])
    nsc = vN.shape[0]

    x = np.asarray(events[obs], float)
    w = np.asarray(events[weight_key], float)
    if window is not None:
        xm, xH = window
        m = (x >= xm) & (x < xH)
        x, w = x[m], w[m]
        a, b = xm, xH
    # SNR needs the moment INTEGRALS, which converge on a subsample; the FO uncertainty,
    # not the event count, sets sigma_FO. Subsample (weight-preserving) for speed so the
    # 200-resample MC stays seconds, not minutes, on multi-million-event priors.
    if n_events_max is not None and len(x) > n_events_max:
        sub = np.random.default_rng(seed).choice(len(x), size=int(n_events_max), replace=False)
        x, w = x[sub], w[sub]
    u = _umap(np.clip(x, a, b), a, b, mp)
    C = _cheb(u, Nmax)                                     # (nev, Nmax+1)
    bidx = np.clip(np.searchsorted(xlo, x) - 1, 0, len(xlo) - 1)   # event -> FO bin

    def mu_of(K):        # K per FO bin -> per-event weights -> moments (orders 1..Nmax)
        wk = w * K[bidx]
        s = wk.sum()
        return np.array([(wk * C[:, n]).sum() / s for n in range(1, Nmax + 1)])

    def Kfrom(vn, vl):   # binned differential K-factor (guarded)
        return np.where((vl > 1e-30) & (vn > 0), vn / np.maximum(vl, 1e-30), 1.0)

    mu_c = mu_of(Kfrom(vN[0], vL[0]))                     # central-scale target
    mu_prior = np.array([(w * C[:, n]).sum() / w.sum() for n in range(1, Nmax + 1)])
    d_mu = mu_c - mu_prior

    # scale component: envelope over the scale rows
    mus = np.array([mu_of(Kfrom(vN[s], vL[s])) for s in range(nsc)])
    sig_scale = 0.5 * (mus.max(0) - mus.min(0))

    # statistical component: MC resample both orders within their per-bin errors
    acc = []
    for _ in range(n_stat):
        vn = np.maximum(vN[0] + rng.normal(0, 1, vN[0].shape) * eN[0], 1e-30)
        vl = np.maximum(vL[0] + rng.normal(0, 1, vL[0].shape) * eL[0], 1e-30)
        acc.append(mu_of(Kfrom(vn, vl)))
    sig_stat = np.std(np.array(acc), axis=0)

    sig = np.sqrt(sig_scale ** 2 + sig_stat ** 2)
    return np.abs(d_mu) / np.maximum(sig, 1e-30)


def resolved_order(snr, threshold=1.0) -> int:
    """Largest moment order n with SNR_n > threshold (0 if none)."""
    snr = np.asarray(snr, float)
    hits = [n for n in range(1, len(snr) + 1) if snr[n - 1] > threshold]
    return max(hits, default=0)


# ---------------------------------------------------------------------------
# Multi-part FO dispatch: let the engine read a per-observable histogram bank.
# (The engine's _fo_hist/_nscales take ONE fo container; we hand them a
#  {'kind':'multi','parts':{obs:dat}} bank and dispatch on obs. Same mechanism
#  the frozen diphoton driver uses; installed once, idempotently.)
# ---------------------------------------------------------------------------
def _install_multi_dispatch():
    if getattr(_mm, "_maxent_upgrade_patched", False):
        return
    _orig_fo_hist = _mm._fo_hist
    _orig_nscales = _mm._nscales

    def _fo_hist(fo, obs, edges, k=0):
        if isinstance(fo, dict) and fo.get("kind") == "multi":
            return _orig_fo_hist(fo["parts"][obs], obs, edges, k)
        return _orig_fo_hist(fo, obs, edges, k)

    def _nscales(fo):
        if isinstance(fo, dict) and fo.get("kind") == "multi":
            return fo["nsc"]
        return _orig_nscales(fo)

    _mm._fo_hist = _fo_hist
    _mm._nscales = _nscales
    _mm._maxent_upgrade_patched = True


_install_multi_dispatch()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class UpgradeResult:
    weights: np.ndarray                 # new positive per-event weights (central solve)
    effN: float                         # effective sample FRACTION in (0, 1]
    closure: float                      # worst relative moment closure |achieved/target - 1|
    x_match: float                      # recoil matching scale (seam) chosen by the engine
    report: Dict[str, Any]              # full engine report + moment-selection block
    moment_snr: Dict[str, np.ndarray]   # per-observable SNR spectrum (orders 1..Nmax)
    chosen_moments: Dict[str, int] = field(default_factory=dict)  # #moments imposed per obs
    band: Optional[Dict[str, np.ndarray]] = None   # variant weights (scales + rate schemes)

    def summary(self) -> str:
        cm = ", ".join(f"{k}={v}" for k, v in self.chosen_moments.items())
        return (f"effN={100*self.effN:.2f}%  closure={self.closure:.1e}  "
                f"x_match={self.x_match:.3g}  moments[{cm}]  "
                f"positive={(self.weights > 0).all()}"
                + (f"  band_variants={len(self.band)}" if self.band else ""))


# ---------------------------------------------------------------------------
# The one public entry point
# ---------------------------------------------------------------------------
def upgrade(events: Dict[str, np.ndarray],
            fo_low: Optional[Dict[str, FOHist]],
            fo_high: Dict[str, FOHist],
            config: Dict[str, Any]) -> UpgradeResult:
    r"""Upgrade a PS+LO (order N) event sample to PS+N^kLO (order N+k).

    Parameters
    ----------
    events : dict of 1D np.ndarray
        The prior event sample.  MUST contain the weight array (key `config['weight_key']`,
        default 'weight'; strictly positive), every Born observable, and the recoil
        observable named in `config`.  Any additional keys are treated as unconstrained
        FOLLOWERS (they inherit the new weights automatically; no FO input needed).
    fo_low, fo_high : dict {obs_name: FOHist}
        The fixed-order differential distributions at the two consecutive orders N and
        N+k, at the SAME fiducial cuts as `events`, for every Born and recoil observable.
        `fo_low` may be None only for the k=1 special case (prior is order N, no order-N
        recoil spectrum); then moment selection falls back to fixed counts.
    config : dict
        Declares each observable's ROLE and variable map/range, plus knobs.  Example::

            config = {
              'born':    {'mll':   {'range': (66., 116.), 'map': 'lin'},
                          'y_abs': {'range': (0., 2.4),   'map': 'lin'}},
              'recoil':  {'pT_ll': {'range': (0.5, 500.), 'map': 'log',
                                    'soft_lo': 0.5}},
              'followers': ['phistar', 'pT_lead'],
              # optional knobs (see DEFAULTS / README): weight_key, moment_selection,
              # snr_threshold, snr_max_order, L2, RELMAX, DELTA, WSIG, CLIP,
              # MINSUP, RSUP, band, born_N, recoil_N
            }

    Returns
    -------
    UpgradeResult
    """
    cfg = {**DEFAULTS, **config}
    wkey = cfg["weight_key"]

    # ---- validate inputs ----------------------------------------------------
    if wkey not in events:
        raise KeyError(f"events is missing the weight key '{wkey}'")
    born_cfg = config.get("born", {})
    recoil_cfg = config.get("recoil", {})
    if len(recoil_cfg) != 1:
        raise ValueError("config['recoil'] must declare exactly ONE recoil observable")
    followers = list(config.get("followers", []))
    w = np.asarray(events[wkey], float)
    if not np.all(w > 0):
        raise ValueError("prior weights must be strictly positive")
    for obs in list(born_cfg) + list(recoil_cfg) + followers:
        if obs not in events:
            raise KeyError(f"events is missing observable array '{obs}'")
    for obs in list(born_cfg) + list(recoil_cfg):
        if obs not in fo_high:
            raise KeyError(f"fo_high is missing FO histogram for constrained observable '{obs}'")

    # engine 'prior' dict: shallow copy of events with the canonical weight key 'w'
    prior = dict(events)
    prior["w"] = w

    recoil_obs = next(iter(recoil_cfg))
    r_lo, r_hi = recoil_cfg[recoil_obs]["range"]
    r_map = recoil_cfg[recoil_obs].get("map", "log")
    soft_lo = recoil_cfg[recoil_obs].get("soft_lo", max(r_lo, 1e-6))

    # ---- assemble the engine's multi-part FO banks --------------------------
    parts_hi = {o: _as_dat(fo_high[o]) for o in list(born_cfg) + [recoil_obs]}
    parts_lo = (None if fo_low is None
                else {o: _as_dat(fo_low[o]) for o in list(born_cfg) + [recoil_obs]})
    nsc = min(len(d["vals"]) for d in parts_hi.values())
    if parts_lo is not None:
        nsc = min(nsc, min(len(d["vals"]) for d in parts_lo.values()))
    FO_HI = {"kind": "multi", "parts": parts_hi, "nsc": nsc}
    FO_LO = None if parts_lo is None else {"kind": "multi", "parts": parts_lo, "nsc": nsc}

    # ---- engine cfg ---------------------------------------------------------
    born_edges = {o: _edges(parts_hi[o]) for o in born_cfg}
    recoil_edges = _edges(parts_hi[recoil_obs])
    ecfg = dict(
        born=[(o, born_cfg[o]["range"][0], born_cfg[o]["range"][1], born_cfg[o].get("map", "lin"))
              for o in born_cfg],
        born_edges=born_edges,
        recoil=(recoil_obs, recoil_edges, soft_lo),
        followers=followers,
    )

    # ---- dials --------------------------------------------------------------
    dials = dict(_DIALS)
    for key in _DIAL_KEYS:
        if key in config:
            dials[key] = config[key]

    # ---- moment selection ---------------------------------------------------
    snr_spectra: Dict[str, np.ndarray] = {}
    chosen: Dict[str, int] = {}
    thr = cfg["snr_threshold"]
    born_ceiling = cfg["snr_max_order"]["born"]
    recoil_ceiling = cfg["snr_max_order"]["recoil"]
    moment_selection = cfg["moment_selection"] and (FO_LO is not None)

    if moment_selection:
        # Born observables: SNR over the full [range]
        for o in born_cfg:
            a, b = born_cfg[o]["range"]
            mp = born_cfg[o].get("map", "lin")
            snr = moment_snr(parts_lo[o], parts_hi[o], events, o, a, b, mp, born_ceiling,
                             weight_key=wkey, n_stat=cfg["snr_n_stat"], seed=cfg["snr_seed"])
            snr_spectra[o] = snr
            chosen[o] = resolved_order(snr, thr)
        # Recoil observable: SNR over the FO window [x_match, x_hi]
        xm_idx = _kconv_xmatch(parts_hi[recoil_obs], parts_lo[recoil_obs], recoil_obs,
                               recoil_edges, dials["DELTA"], {}, relmax=dials["RELMAX"])
        x_match_snr = float(recoil_edges[xm_idx])
        x_hi_snr = float(min(recoil_edges[-1], np.asarray(events[recoil_obs], float).max()))
        snr = moment_snr(parts_lo[recoil_obs], parts_hi[recoil_obs], events, recoil_obs,
                         r_lo, r_hi, r_map, recoil_ceiling, weight_key=wkey,
                         window=(x_match_snr, x_hi_snr),
                         n_stat=cfg["snr_n_stat"], seed=cfg["snr_seed"])
        snr_spectra[recoil_obs] = snr
        chosen[recoil_obs] = resolved_order(snr, thr)
        ecfg["born_N"] = {o: chosen[o] for o in born_cfg}
        ecfg["recoil_N"] = max(chosen[recoil_obs], 1)   # need >=1 recoil moment to shape the window
    else:
        # fixed counts (config-supplied or engine defaults)
        fixed_born = config.get("born_N")
        fixed_recoil = config.get("recoil_N")
        if fixed_born is not None:
            ecfg["born_N"] = dict(fixed_born)
        if fixed_recoil is not None:
            ecfg["recoil_N"] = int(fixed_recoil)
        for o in born_cfg:
            chosen[o] = int((fixed_born or {}).get(o, dials["N_FO"])) if fixed_born else dials["N_FO"]
        chosen[recoil_obs] = int(fixed_recoil) if fixed_recoil is not None else dials["N_W"]

    # ---- solve --------------------------------------------------------------
    report: Dict[str, Any] = {}
    if cfg["band"] and FO_LO is not None:
        variants, report = _band_solve(prior, ecfg, FO_LO, FO_HI, dials=dials)
        q = variants["central"]
        band = {k: v for k, v in variants.items() if k != "central"}
    else:
        q, report = _build_and_solve(prior, ecfg, FO_LO, FO_HI, 0, "max", dials, None, report)
        band = None
    if q is None:
        raise RuntimeError("MaxEnt solve did not converge to a positive-weight solution; "
                           "inspect report for the failing constraint")

    # ---- assemble result ----------------------------------------------------
    report = dict(report)
    report.pop("_lam", None)
    report["moment_selection"] = dict(
        enabled=bool(moment_selection),
        threshold=thr,
        chosen=dict(chosen),
        snr={o: [float(s) for s in snr_spectra[o]] for o in snr_spectra},
        hit_ceiling={o: (chosen[o] >= (recoil_ceiling if o == recoil_obs else born_ceiling))
                     for o in snr_spectra},
    )
    solve = report.get("solve", {})
    effN = float(solve.get("effN_pct", np.nan)) / 100.0
    closure = float(solve.get("worst_rel_closure", np.nan))
    x_match = float(report.get("kconv", {}).get("xm_edge", np.nan))

    return UpgradeResult(
        weights=np.asarray(q, float),
        effN=effN,
        closure=closure,
        x_match=x_match,
        report=report,
        moment_snr=snr_spectra,
        chosen_moments=dict(chosen),
        band=band,
    )
