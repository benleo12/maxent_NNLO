#!/usr/bin/env python3
r"""maxent_upgrade -- positive-weight event samples carrying fixed-order (NLO/NNLO) information.

    from maxent_upgrade import upgrade, load_moments
    moments, config = load_moments("dy")          # released event-level moments (or your own)
    result = upgrade(events, moments, config)    # events: your positive-weight LO+PS prior
    print(result.describe())

`upgrade` reweights the prior p_i by the maximum-entropy exponential tilt
q_i = p_i exp(sum_a lambda_a m_a(Phi_i)) / Z that reproduces the fixed-order event-level
Chebyshev moments: Born towers over the full fiducial range, recoil towers inside a smooth
window above the matching scale.  The weights are positive by construction; every observable
that is not constrained (a follower) inherits them.  See README.md for the contract.

Public API
----------
    upgrade(events, moments, config) -> UpgradeResult    # event-level moments (the method)
    load_moments(process, ...) -> (moments, config)      # the paper's released tables
    sigma_fid(process) -> dict or None                   # their fiducial cross sections
    released                                             # table(), summary(), processes()
    compute_fo_moments(fo_events, config, x_match, ...)  # moments from FO events in Python
    chebyshev_moment(x, w, n_max, a, b, mp)              # the moment definition itself
    profile_w(x, a, b, c=None, d=None)                   # the C^2 window
    matching_scale(Q), check_seam(x_match, Q)            # the matching-scale rule
    combine_moments(anchor, target, n=None)              # for soft_constraints=True
    UpgradeResult, DEFAULTS
Legacy (an earlier, histogram-based variant; not used for the papers' results):
    upgrade_from_histograms(events, fo_low, fo_high, config), moment_snr, resolved_order,
    FOHist, fo_from_dat
"""
import os
import warnings
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
    _newton_maxent as _newton,
)
from .basis import cheb as _cheb, umap as _umap
from .released import load_moments, sigma_fid          # the paper's fixed-order moments
from . import released                                 # released.table(), .summary(), .processes()

__version__ = "1.1.0"

__all__ = [
    "upgrade", "load_moments", "sigma_fid", "released",
    "upgrade_from_histograms", "compute_fo_moments", "chebyshev_moment",
    "profile_w", "matching_scale", "check_seam",
    "moment_snr", "resolved_order", "combine_moments",
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
    # --- band variants: impose the CENTRAL solve's constraint set ---------------
    fixed_orders=None,       # {obs: N}: impose exactly N orders of that tower (overrides the SNR rule)
    # --- Gaussian soft constraints (see README, "Soft constraints") -----------
    soft_constraints=False,  # impose each moment with a Gaussian penalty of width = its error
    soft_errors="stat",      # width: "stat" = Monte Carlo error of the target; "total" = stat (+) scale
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
    """What `upgrade()` returns.

    Attributes
    ----------
    weights : np.ndarray
        The new per-event weights q_i, strictly positive, summing to one.
    effN : float
        Kish effective sample fraction 1/(N sum q_i^2), in (0, 1].
    effN_retained : float
        Fraction of the PRIOR's own Kish count that the upgrade keeps,
        Kish(q)/Kish(p).  Equals effN for an unweighted prior; for a weighted (e.g.
        sliced) prior it is the meaningful cost of the upgrade.
    closure : float
        Worst relative deviation of an achieved moment from its target, max|ach/mu - 1|.
        Near-zero targets inflate it; see max_pull.
    max_pull : float
        Worst |achieved - target| / sigma over the constraints, with sigma the target's
        Monte Carlo (stat) error where the moments carry `stat_errors`, otherwise its total
        error.
    x_match : float
        Matching scale of the first recoil window (nan without one).
    moment_snr : dict {observable: array}
        Resolution spectrum SNR_n = |mu_n(target) - mu_n(prior)| / sigma_n, orders 1..N.
    chosen_moments : dict {observable or joint key: int}
        Number of moments imposed per tower (towers are imposed up to their highest
        resolved order) and per joint block.
    report : dict
        Multipliers `lam`, targets `mu`, `feature_names`, `towers` (per-tower table used by
        describe()), `moment_selection`, `windows`, the soft-constraint block, and with
        `keep_features=True` the feature matrix `Phi` and prior `p`.
    band : None
        Reserved; bands are re-solves of upgrade() (see README, "Uncertainty bands").
    soft_pulls : np.ndarray or None
        soft_constraints=True only: (achieved - target)/sigma per constraint.
    """
    weights: np.ndarray                 # new positive per-event weights (central solve)
    effN: float                         # effective sample FRACTION in (0, 1]
    closure: float                      # worst relative moment closure |achieved/target - 1|
    x_match: float                      # matching scale of the first recoil window
    report: Dict[str, Any]              # multipliers, targets, per-tower table, moment selection
    moment_snr: Dict[str, np.ndarray]   # per-observable SNR spectrum (orders 1..Nmax)
    chosen_moments: Dict[str, int] = field(default_factory=dict)  # #moments imposed per tower
    band: Optional[Dict[str, np.ndarray]] = None   # legacy histogram path only
    soft_pulls: Optional[np.ndarray] = None        # soft mode: (achieved - target)/sigma per constraint
    effN_retained: float = float("nan")            # Kish(q)/Kish(p)
    max_pull: float = float("nan")                 # worst |achieved - target|/sigma

    def summary(self) -> str:
        """One line: effN, closure, worst pull, moments imposed, positivity."""
        cm = ", ".join(f"{k}={v}" for k, v in self.chosen_moments.items())
        return (f"effN={100*self.effN:.2f}%  closure={self.closure:.1e}"
                + (f"  max_pull={self.max_pull:.2f}" if np.isfinite(self.max_pull) else "")
                + f"  x_match={self.x_match:.3g}  moments[{cm}]  "
                f"positive={(self.weights > 0).all()}"
                + (f"  band_variants={len(self.band)}" if self.band else "")
                + (f"  worst_soft_pull={np.nanmax(np.abs(self.soft_pulls)):.2f}"
                   if self.soft_pulls is not None and np.isfinite(self.soft_pulls).any() else ""))

    def describe(self) -> str:
        """A table of what was imposed and how well it closed, one row per tower."""
        towers = self.report.get("towers") or []
        out = [f"{'tower':16s} {'kind':6s} {'map':4s} {'imposed':>9s}  {'orders below SNR 1 (SNR)':28s} {'max pull':>8s}"]
        for t in towers:
            below = ", ".join(f"{n}({s:.2f})" for n, s in t.get("below", [])) or "-"
            mp = t.get("max_pull", float("nan"))
            out.append(f"{t['name']:16s} {t['kind']:6s} {str(t.get('map', '-'))[:4]:4s} "
                       f"{t['imposed']:>4d} / {t['supplied']:<3d} {below[:28]:28s} "
                       f"{(f'{mp:8.3f}' if np.isfinite(mp) else '       -')}")
        qp = self.report.get("q_over_p_range", (float("nan"), float("nan")))
        out.append(f"N = {len(self.weights):,}   effN = {100*self.effN:.2f}%   "
                   f"retained of the prior's effective count = {100*self.effN_retained:.2f}%")
        out.append(f"closure (max rel.) = {self.closure:.2e}   max pull = {self.max_pull:.3f}   "
                   f"q/p in [{qp[0]:.3g}, {qp[1]:.3g}]   negative weights: {int((self.weights <= 0).sum())}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# The one public entry point
# ---------------------------------------------------------------------------
def upgrade_from_histograms(events: Dict[str, np.ndarray],
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
              'recoil':  {'pT_ll': {'range': (30., 2500.), 'map': 'log',
                                    'soft_lo': 30.}},
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
    if not recoil_cfg and not config.get("mixed"):
        raise ValueError("config must declare at least one recoil or mixed constraint")
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


# ===========================================================================
# EVENT-LEVEL MOMENT INTERFACE (the method of the papers).  The fixed-order
# input is the moments themselves, weighted sums over the fixed-order phase-space
# points, never a re-binned histogram.
# ===========================================================================
def chebyshev_moment(x, w, n_max, a, b, mp):
    r"""Event-level Chebyshev moments <T_1..T_n_max> of observable x under weights w.

        mu_n = sum_i w_i T_n(u(x_i)) / sum_i w_i ,   u = umap(x, a, b, mp)

    This is the weighted sum a fixed-order code performs when it books the observable
    T_n(u(x)); there is no binning.  `mp` is 'lin', 'log', 'bw' or ('bw', M, Gamma);
    values outside [a, b] are clipped to the edge, as in the NNLOJET patch.
    """
    x = np.asarray(x, float); w = np.asarray(w, float)
    u = _umap(np.clip(x, a, b), a, b, mp)
    C = _cheb(u, n_max)
    s = w.sum()
    return np.array([(w * C[:, n]).sum() / s for n in range(1, n_max + 1)])


def alpha_s(mu, mz=91.1876, az=0.118, nf=5):
    """Two-loop running coupling, normalised at m_Z."""
    mu = np.asarray(mu, float)
    b0 = (33 - 2 * nf) / (12 * np.pi); b1 = (153 - 19 * nf) / (24 * np.pi ** 2)
    L = np.log(np.maximum(mu, 1.0) ** 2 / mz ** 2); al = az
    for _ in range(60):
        al = az / (1 + az * b0 * L + az ** 2 * (b1 / b0)
                   * np.log(np.maximum(1 + az * b0 * L, 1e-9)))
    return al


def matching_scale(Q, thr=0.2, lo=3.0, hi=None, n=4000):
    r"""The matching scale v_m for a process with hard scale Q, from power counting:

        v_m = min { pT :  alpha_s(pT) ln^2(Q/pT)  <  thr }      (thr = 0.2)

    Below it unresummed Sudakov logarithms spoil the fixed-order series and the shower
    is kept; above it the fixed order is imposed.  The process enters only through Q:
    27.9 GeV for Q = m_Z, 24.9 GeV for Q = m_W, 27.6 GeV for Q = 90 GeV (diphoton),
    37.1 GeV for Q = m_H.  Returns the scale in GeV.
    """
    Q = float(Q)
    hi = 0.9 * Q if hi is None else float(hi)
    pt = np.geomspace(lo, hi, int(n))
    v = alpha_s(pt) * np.log(Q / np.minimum(pt, 0.99 * Q)) ** 2
    ok = np.where(v < thr)[0]
    if not len(ok):
        raise ValueError(f"no pT in [{lo}, {hi}] satisfies alpha_s ln^2 < {thr} for Q={Q}")
    return float(pt[ok[0]])


def check_seam(x_match, Q, tol=0.15, label="", thr=0.2):
    r"""Warn when a matching scale disagrees with `matching_scale(Q)` by more than `tol`.

    The matching scale is not a free parameter of the reweighting: the window is
    compiled into the fixed-order run (NNLOJET `eval_w_*`), and the targets are the
    windowed moments <w T_n>/<w> with that window, so `x_match` must mirror it.
    Changing it on the Python side alone imposes moments built with one window
    against features built with another.  This function therefore only warns (it
    prints); moving the matching scale means re-running the fixed order.
    `load_moments` calls it for every recoil window of a released table.

    Returns the rule's value in GeV.
    """
    want = matching_scale(Q, thr=thr)
    if abs(x_match - want) / want > tol:
        print(f"  [matching scale] {label or 'process'}: the window starts at x_match = {x_match:.1f} GeV, "
              f"but alpha_s ln^2(Q/pT) < {thr} at Q = {Q:g} GeV gives {want:.1f} GeV "
              f"({100*(x_match-want)/want:+.0f}%).  The window is compiled into the fixed-order "
              f"run: to adopt {want:.1f} GeV, change it there and regenerate the moments.")
    return want


def _smootherstep(t):
    """C2 smootherstep 6t^5-15t^4+10t^3, clipped to [0,1]."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def profile_w(x, a, b, c=None, d=None):
    r"""The C^2 window w(x) in ln(x) (the paper's eq. for the recoil window).

    w = 0 for x <= a, rises as 6t^5 - 15t^4 + 10t^3 with t = ln(x/a)/ln(b/a) to 1 at b,
    stays 1, then either stops at a hard cut at c (d None) or falls C^2 to 0 between
    c and d; no upper edge if c is None.  a == b gives a hard lower step.
    The released tables give (a, b, c[, d]) as `profile`; `config` uses the same key.
    """
    x = np.asarray(x, float)
    lx = np.log(np.maximum(x, 1e-30))
    up = (_smootherstep((lx - np.log(a)) / (np.log(b) - np.log(a)))
          if b > a else (x >= a).astype(float))
    if c is None:
        return up
    if d is None or d <= c:
        down = (x < c).astype(float)
    else:
        down = 1.0 - _smootherstep((lx - np.log(c)) / (np.log(d) - np.log(c)))
    return up * down


def compute_fo_moments(fo_events, config, x_match=None, x_hi=None, n_max=None,
                       weight_key="weight", scale_key="weight_scales"):
    r"""Build the `moments` argument of `upgrade` from fixed-order EVENTS held in Python.

    Use this only if you have the fixed-order phase-space points in memory; otherwise
    accumulate the same weighted sums inside your fixed-order code (MOMENT_CONSTRUCTION.md)
    and fill the dict by hand.  `fo_events` is a dict of 1D arrays: every constrained
    observable, the weights under `weight_key`, and optionally a (n_scale, n_event) array
    under `scale_key` with the weight of each scale choice (row 0 = central).

    Born towers are taken over the full declared range.  Each recoil tower uses the window
    of its `config['recoil'][obs]['profile']` (a, b, c[, d]) when one is given -- the
    windowed moments <w T_n>/<w> and the window rate <w>/<1>, as the solver imposes them --
    and otherwise the hard window [x_match, x_hi).  Errors: `stat_errors` is the weighted
    Monte Carlo error, `scale_errors` the half-range over the scale rows, `errors` (and
    `window_errors`) their quadrature sum.
    """
    born_cfg = config.get("born", {}); recoil_cfg = config.get("recoil", {})
    w = np.asarray(fo_events[weight_key], float)
    wsc = fo_events.get(scale_key)
    wsc = None if wsc is None else np.atleast_2d(np.asarray(wsc, float))
    Nb = n_max if n_max is not None else DEFAULTS["snr_max_order"]["born"]
    Nr = n_max if n_max is not None else DEFAULTS["snr_max_order"]["recoil"]

    def stat_err(x, ww, nmax, a, b, mp):
        # weighted-MC error on each moment: sqrt(sum w^2 (T_n - mu_n)^2)/sum w
        u = _umap(np.clip(np.asarray(x, float), a, b), a, b, mp); C = _cheb(u, nmax)
        s = ww.sum(); mu = np.array([(ww * C[:, n]).sum() / s for n in range(1, nmax + 1)])
        return np.array([np.sqrt((ww**2 * (C[:, n] - mu[n-1])**2).sum()) / s
                         for n in range(1, nmax + 1)])

    def scale_env(x, nmax, a, b, mp):
        if wsc is None:
            return None
        vals = np.array([chebyshev_moment(x, wsc[r], nmax, a, b, mp) for r in range(wsc.shape[0])])
        return 0.5 * (vals.max(0) - vals.min(0))

    def pack(mu, se, sc):
        d = dict(stat_errors=se.tolist(),
                 errors=(se if sc is None else np.sqrt(se**2 + sc**2)).tolist())
        if sc is not None:
            d["scale_errors"] = sc.tolist()
        return d

    moments = {"born": {}, "recoil": {}}
    for o in born_cfg:
        a, b = born_cfg[o]["range"]; mp = born_cfg[o].get("map", "lin")
        x = np.asarray(fo_events[o], float)
        mu = chebyshev_moment(x, w, Nb, a, b, mp)
        moments["born"][o] = dict(values=mu.tolist(), **pack(mu, stat_err(x, w, Nb, a, b, mp),
                                                             scale_env(x, Nb, a, b, mp)))

    for obs, rc in recoil_cfg.items():
        xr = np.asarray(fo_events[obs], float)
        soft_lo = float(rc.get("soft_lo", max(rc["range"][0], 1e-6)))
        xhi = float(x_hi if x_hi is not None else rc["range"][1])
        mp = rc.get("map", "log")
        prof = rc.get("profile")
        if prof:
            a = float(prof.get("a", x_match)); b = float(prof.get("b", a))
            c = prof.get("c", xhi); d = prof.get("d", None)
            if x_match is not None and not np.isclose(float(x_match), a):
                warnings.warn(f"compute_fo_moments: {obs} has a window starting at a={a:g}; x_match={x_match:g} is ignored")
            wf = profile_w(xr, a, b, None if c is None else float(c), None if d is None else float(d))
            xm = a
        else:
            if x_match is None:
                raise ValueError(f"compute_fo_moments: {obs} has no 'profile' in config, so x_match is needed")
            wf = ((xr >= x_match) & (xr < xhi)).astype(float); xm = float(x_match)
        sel = wf > 0
        ww = w * wf
        muw = chebyshev_moment(xr[sel], ww[sel], Nr, soft_lo, xhi, mp)
        sc = (None if wsc is None else
              0.5 * np.ptp([chebyshev_moment(xr[sel], wsc[r][sel] * wf[sel], Nr, soft_lo, xhi, mp)
                            for r in range(wsc.shape[0])], axis=0))
        dd = pack(muw, stat_err(xr[sel], ww[sel], Nr, soft_lo, xhi, mp), sc)
        R = float(ww.sum() / w.sum())
        R_stat = float(np.sqrt(((w * (wf - R)) ** 2).sum()) / w.sum())
        R_scale = (0.0 if wsc is None else
                   float(0.5 * np.ptp([(wsc[r] * wf).sum() / wsc[r].sum() for r in range(wsc.shape[0])])))
        moments["recoil"][obs] = dict(
            window_values=muw.tolist(), window_errors=dd["errors"],
            stat_errors=dd["stat_errors"], **({"scale_errors": dd["scale_errors"]} if "scale_errors" in dd else {}),
            wprofile_values=muw.tolist(), wprofile_rate=R,
            rate=R, rate_stat=R_stat, rate_scale=R_scale,
            x_match=xm, x_hi=xhi, soft_lo=soft_lo)
    return moments


# config keys upgrade() reads, and the legacy keys only upgrade_from_histograms() reads
_UPGRADE_KEYS = {"born", "recoil", "mixed", "followers", "weight_key", "moment_selection",
                 "snr_threshold", "L2", "fixed_orders", "lam0", "keep_features",
                 "soft_constraints", "soft_errors"}
_HIST_ONLY_KEYS = {"band", "RELMAX", "DELTA", "WSIG", "CLIP", "MINSUP", "RSUP", "born_N",
                   "recoil_N", "snr_max_order", "snr_n_stat", "snr_seed"}


def _check_inputs(events, moments, config, wkey):
    """Fail early, with a message that says what to fix, on inconsistent inputs."""
    for k in config:
        if k in _HIST_ONLY_KEYS:
            warnings.warn(f"config['{k}'] is read only by the legacy upgrade_from_histograms(); "
                          f"upgrade() ignores it", stacklevel=3)
        elif k not in _UPGRADE_KEYS:
            warnings.warn(f"config['{k}'] is not a key upgrade() knows; it is ignored "
                          f"(known: {sorted(_UPGRADE_KEYS)})", stacklevel=3)
    born_cfg = config.get("born", {}) or {}; recoil_cfg = config.get("recoil", {}) or {}
    mixed_cfg = config.get("mixed", {}) or {}
    if not recoil_cfg and not mixed_cfg and not born_cfg:
        raise ValueError("config declares no constraint: give at least one of 'born', 'recoil', 'mixed'")
    need = list(born_cfg) + list(recoil_cfg) + [o for m in mixed_cfg.values() for o in m["observables"]]
    need = list(dict.fromkeys(need))
    missing = [k for k in need + [wkey] if k not in events]
    if missing:
        raise KeyError(f"events lacks {missing}. This config needs one array per constrained observable "
                       f"{need} plus the prior weights '{wkey}'; events has {sorted(events)}")
    n = len(np.asarray(events[wkey]))
    bad = {k: len(np.asarray(events[k])) for k in need if len(np.asarray(events[k])) != n}
    if bad:
        raise ValueError(f"events arrays differ in length: '{wkey}' has {n}, but {bad}")
    for k in need + [wkey]:
        nf = int((~np.isfinite(np.asarray(events[k], float))).sum())
        if nf:
            raise ValueError(f"events['{k}'] has {nf} non-finite entries; remove those events first")
    for role, keys in (("born", born_cfg), ("recoil", recoil_cfg), ("mixed", mixed_cfg)):
        for k in keys:
            if k not in (moments.get(role) or {}):
                raise KeyError(f"config['{role}'] names '{k}', but moments['{role}'] has no entry for it "
                               f"(has {sorted(moments.get(role) or {})})")
    for o, bc in born_cfg.items():
        a, b = bc["range"]; x = np.asarray(events[o], float)
        out = float(np.mean((x < a) | (x > b)))
        if out > 0.05:
            warnings.warn(f"{100*out:.1f}% of the prior's '{o}' lies outside the tower range [{a:g}, {b:g}]; "
                          f"those events are clipped to the edge.  Same fiducial cuts and units on both sides?",
                          stacklevel=3)
    for o, rc in recoil_cfg.items():
        mrc = moments["recoil"][o]; x = np.asarray(events[o], float)
        prof = rc.get("profile")
        if prof:
            win = profile_w(x, float(prof.get("a", mrc["x_match"])), float(prof.get("b", mrc["x_match"])),
                            None if prof.get("c") is None else float(prof["c"]),
                            None if prof.get("d") is None else float(prof["d"])) > 0
        else:
            win = (x >= float(mrc["x_match"])) & (x < float(mrc.get("x_hi", np.inf)))
        nin = int(win.sum())
        if nin == 0:
            raise ValueError(f"no prior event lies inside the '{o}' window (from {float(mrc['x_match']):g}); "
                             f"check the units (GeV) and the fiducial selection of events['{o}']")
        if nin < 100:
            warnings.warn(f"only {nin} prior events lie inside the '{o}' window; the upgrade there is "
                          f"Monte Carlo noise times a tilt", stacklevel=3)


def upgrade(events, moments, config):
    r"""Reweight a positive-weight prior so that it reproduces event-level fixed-order moments.

    Parameters
    ----------
    events : dict of 1D np.ndarray, all the same length
        The prior: the strictly positive weights under 'weight' (or `config['weight_key']`)
        and one array per observable named in `config`.  Every other array is a follower:
        it is not used here and inherits `result.weights`.  No cuts are applied; prior and
        moments must integrate the same phase space.
    moments : dict
        The fixed-order targets, accumulated event by event (never a histogram)::

            moments = {
              'born':   {obs: {'values': [mu_1, ..], 'errors': [s_1, ..],
                               'stat_errors': [..], 'scale_errors': [..]}},     # last two optional
              'recoil': {obs: {'window_values': [..], 'window_errors': [..],
                               'rate': r, 'x_match': a, 'x_hi': xh, 'soft_lo': s0,
                               'stat_errors': [..], 'rate_stat': .., 'rate_scale': ..}},
              'mixed':  {key: {'values': {(m, n): mu_mn}, 'errors': {..}, 'rate': r}},  # optional
            }

        Born: values[n-1] = sum_j w_j T_n(u(x_j)) / sum_j w_j over the fixed-order points.
        Recoil: the windowed moments <w T_n(u)>/<w> with u the log map on [soft_lo, x_hi]
        and w the window of `config['recoil'][obs]['profile']` (or the hard window
        [x_match, x_hi) without a profile), and the window rate r = <w>/<1>.  `errors` are
        the total fixed-order uncertainties (Monte Carlo scatter combined with the scale
        envelope); they decide which orders are resolved.  `load_moments(process)` returns
        this dict for the paper's processes; `compute_fo_moments` builds it from FO events.
    config : dict
        Roles, maps and windows, plus optional knobs::

            config = {
              'born':   {'mll':   {'range': (66., 116.), 'map': 'bw'},   # 'lin', 'log', 'bw'
                         'y_abs': {'range': (0., 2.4),   'map': 'lin'}},
              'recoil': {'pT_ll': {'range': (30., 2500.), 'map': 'log', 'soft_lo': 30.,
                                   'profile': {'a': 30., 'b': 60., 'c': 500.}}},
            }

        Knobs (defaults in DEFAULTS): moment_selection=True (impose each tower up to its
        highest order with SNR > snr_threshold=1.0), L2=1e-4 (ridge of the dual), fixed_orders
        ({obs: N}: impose exactly N orders, e.g. for band variants), lam0 (warm start from a
        previous result.report['lam']), keep_features=False (return Phi and p in the report),
        soft_constraints=False and soft_errors='stat' (Gaussian soft constraints),
        weight_key='weight'.  Unknown keys raise a warning.

    Returns
    -------
    UpgradeResult (see its docstring): weights, effN, effN_retained, closure, max_pull,
    moment_snr, chosen_moments, x_match, report; result.describe() prints a summary table.
    """
    cfg = {**DEFAULTS, **config}
    wkey = cfg["weight_key"]
    born_cfg = config.get("born", {}); recoil_cfg = config.get("recoil", {})
    if wkey not in events:
        raise KeyError(f"events lacks the prior weights '{wkey}' (events has {sorted(events)})")
    w = np.asarray(events[wkey], float)
    if not np.all(w > 0):
        raise ValueError(f"prior weights must be strictly positive: {int((w <= 0).sum())} of {len(w)} are not "
                         f"(fold or remove negative weights before the upgrade)")
    _check_inputs(events, moments, config, wkey)
    p = w / w.sum()
    thr = cfg["snr_threshold"]; do_sel = cfg["moment_selection"]

    F = [np.ones(len(w))]; mu = [1.0]; names = ["norm"]
    SIG = [0.0]      # per-constraint target error (feature units); used only in soft mode
    snr_spectra = {}; chosen = {}; towers = []     # towers: one row per tower for describe()
    soft = bool(cfg["soft_constraints"])

    def _soft_err(d, total_key):
        """Per-order width for the soft penalty: the Monte Carlo error by default."""
        st = d.get("stat_errors") if cfg["soft_errors"] == "stat" else None
        if st is not None and len(st):
            return np.asarray(st, float)
        tot = d.get(total_key)
        return np.asarray(tot, float) if tot is not None else np.zeros(0)

    def _rate_err(d):
        st = float(d.get("rate_stat", 0.0) or 0.0)
        if cfg["soft_errors"] == "total":
            st = float(np.hypot(st, float(d.get("rate_scale", 0.0) or 0.0)))
        return st

    # ---- ONE rule for every observable ------------------------------------
    # Impose the FO moments inside the observable's validity window [XM, XHI];
    # preserve the prior below XM and above XHI. Each constrained observable is
    # a single call to _impose below. The ONLY per-observable difference is the
    # validity window: for a Born variable the fixed order is valid over the
    # whole fiducial range (XM=-inf, XHI=+inf, rate=1, nothing preserved, so the
    # composite reduces to imposing the FO moment); for the recoil it is the
    # window [x_match, x_hi] and the shower is preserved outside. There is no
    # separate "recoil procedure".
    def _impose(obs, lo_map, hi_map, mp, XM, XHI, vals, errs, rate, floor, wprof=None,
                sig=None, rate_sig=0.0):
        XL = np.asarray(events[obs], float)
        Nmax = len(vals)
        sigv = np.zeros(Nmax) if sig is None or not len(sig) else np.asarray(sig, float)[:Nmax]
        if len(sigv) < Nmax:
            sigv = np.concatenate([sigv, np.zeros(Nmax - len(sigv))])
        # The prior's moment in the resolution criterion must be the same quantity as the
        # target: for a windowed tower the window-weighted <w T_n>/<w> (the target is
        # booked that way in the fixed-order code), for a Born tower or a hard window the
        # plain average over [XM, XHI).
        if wprof is not None:
            wpr = np.asarray(wprof, float); win = wpr > 0
            mu_prior_win = (chebyshev_moment(XL[win], (w * wpr)[win], Nmax, lo_map, hi_map, mp)
                            if win.any() else np.zeros(Nmax))
        else:
            win = (XL >= XM) & (XL < XHI)
            mu_prior_win = (chebyshev_moment(XL[win], w[win], Nmax, lo_map, hi_map, mp)
                            if win.any() else np.zeros(Nmax))
        snr = (np.abs(vals - mu_prior_win) / np.maximum(errs, 1e-30)
               if np.asarray(errs).any() else np.full(Nmax, np.inf))
        snr_spectra[obs] = snr
        N = max(resolved_order(snr, thr) if do_sel else Nmax, floor)
        # a band variant (scale choice, bootstrap replica) must impose the SAME
        # constraint set as the central solve: near the SNR threshold the rule can
        # flip an order in or out between variants, which would change the number
        # of multipliers and make the variant a different problem, not a shifted one
        fixed = (cfg.get("fixed_orders") or {}).get(obs)
        if fixed is not None:
            N = min(max(int(fixed), floor), Nmax)
        chosen[obs] = N
        towers.append(dict(name=obs, kind="born" if floor == 0 else "window", map=mp if isinstance(mp, str) else mp[0],
                           supplied=int(Nmax), imposed=int(N),
                           below=[(n, float(snr[n - 1])) for n in range(1, N + 1) if snr[n - 1] <= thr]))
        u = _umap(np.clip(XL, lo_map, hi_map), lo_map, hi_map, mp); C = _cheb(u, max(Nmax, 1))
        if wprof is None:
            # -------- composite (unmasked) target: legacy path --------
            I_S = XL < XM; I_T = XL >= XHI
            P_tail = float(p[I_T].sum()); P_soft = 1.0 - rate - P_tail
            wS = float(p[I_S].sum())
            for n in range(1, N + 1):
                soft_mom = float((p[I_S] * C[I_S, n]).sum()) / wS if wS > 0 else 0.0
                tail_mom = float((p[I_T] * C[I_T, n]).sum()) / P_tail if P_tail > 0 else 0.0
                fo_mom = float(vals[n - 1])
                F.append(C[:, n]); mu.append(P_soft * soft_mom + rate * fo_mom + P_tail * tail_mom)
                names.append(f"{obs}_T{n}"); SIG.append(rate * float(sigv[n - 1]))
        else:
            # -------- profiled features: exact shower preservation --------
            # feature C[:,n]*w with target <T_n>_FO(w-weighted) * R, plus an
            # explicit rate constraint  sum q*w = R.  Where w=0 (below the seam,
            # above x_hi) the feature vanishes -> those events are untilted by the
            # recoil -> the prior shape is preserved to machine precision.  `vals`
            # here are the w-weighted FO moments; `rate` is the FO w-rate R.
            R = float(rate)
            wprof = np.asarray(wprof, float)
            for n in range(1, N + 1):
                F.append(C[:, n] * wprof); mu.append(float(vals[n - 1]) * R)
                names.append(f"{obs}_T{n}")
                SIG.append(float(np.hypot(float(sigv[n - 1]) * R, float(vals[n - 1]) * float(rate_sig))))
            F.append(wprof.copy()); mu.append(R); names.append(f"{obs}_rate"); SIG.append(float(rate_sig))

    # Born observables: validity is the full fiducial range.
    for o in born_cfg:
        a, b = born_cfg[o]["range"]; mp = born_cfg[o].get("map", "lin")
        _impose(o, a, b, mp, -np.inf, np.inf,
                np.asarray(moments["born"][o]["values"], float),
                np.asarray(moments["born"][o].get("errors", []), float),
                rate=1.0, floor=0, sig=_soft_err(moments["born"][o], "errors"))
    # Recoil observables: validity is the window [x_match, x_hi]; shower kept
    # outside.  There may be MORE THAN ONE -- e.g. the diphoton upgrade
    # constrains both pT_gg and pi-dphi_gg, because constraining pT alone leaves
    # the shower's pT<->dphi correlation wrong and dphi overshoots by ~60% near
    # 2.2 rad.  Each carries its own compiled profile and its own rate
    # constraint; their masked features simply add to the same feature matrix.
    windows = {}
    for recoil_obs in recoil_cfg:
      rc = moments["recoil"][recoil_obs]
      windows[recoil_obs] = dict(x_match=float(rc["x_match"]),
                                 x_hi=float(rc.get("x_hi", np.inf)),
                                 rate=float(rc["rate"]))
      soft_lo = float(rc.get("soft_lo", recoil_cfg[recoil_obs].get("soft_lo",
                      max(recoil_cfg[recoil_obs]["range"][0], 1e-6))))
      XHI = float(rc.get("x_hi", np.asarray(events[recoil_obs], float).max()))
      prof = recoil_cfg[recoil_obs].get("profile")
      if prof:
        # smooth (or hard, if a==b) profile: exact shower preservation below the
        # seam via masked features + explicit rate constraint.  `vals`/`rate` must
        # be the w-weighted FO moments + w-rate; fall back to the window values
        # (exact when the profile is the hard window a=b=x_match, c=d=x_hi).
        xm = float(rc["x_match"])
        a = float(prof.get("a", xm)); b = float(prof.get("b", xm))
        c = prof.get("c", XHI); d = prof.get("d", None)
        wprof = profile_w(np.asarray(events[recoil_obs], float), a, b,
                          None if c is None else float(c), None if d is None else float(d))
        vals = np.asarray(rc.get("wprofile_values", rc["window_values"]), float)
        R = float(rc.get("wprofile_rate", rc["rate"]))
        _impose(recoil_obs, soft_lo, XHI, recoil_cfg[recoil_obs].get("map", "log"),
                xm, XHI, vals,
                np.asarray(rc.get("window_errors", []), float),
                rate=R, floor=1, wprof=wprof,
                sig=_soft_err(rc, "window_errors"), rate_sig=_rate_err(rc))
      else:
        _impose(recoil_obs, soft_lo, XHI, recoil_cfg[recoil_obs].get("map", "log"),
                float(rc["x_match"]), XHI,
                np.asarray(rc["window_values"], float),
                np.asarray(rc.get("window_errors", []), float),
                rate=float(rc["rate"]), floor=1,
                sig=_soft_err(rc, "window_errors"), rate_sig=_rate_err(rc))

    # ---- MIXED (two-observable) constraints -------------------------------
    # Separate moments of x and y constrain the two MARGINALS and say nothing
    # about their JOINT distribution.  When a follower is determined by the
    # correlation rather than by either spectrum alone, only a mixed moment can
    # reach it -- constraining the diphoton recoil alone, for instance, left the
    # shower's pT<->dphi correlation wrong.  The feature is
    #     T_m(u(x)) T_n(v(y)) w(x) w(y)
    # with target <T_m T_n>_w * R_xy, plus the joint rate constraint sum q w w.
    # Both profiles multiply, so the feature vanishes unless BOTH observables
    # are inside their windows.
    for key, mc in (config.get("mixed") or {}).items():
        ox, oy = mc["observables"]
        ax, bx = mc["range"][ox]; ay, by = mc["range"][oy]
        mpx = mc.get("map", {}).get(ox, "log"); mpy = mc.get("map", {}).get(oy, "log")
        X = np.asarray(events[ox], float); Y = np.asarray(events[oy], float)
        # A missing or None profile means NO window: the feature is the bare
        # product of Chebyshev polynomials, which is what NNLOJET books for a
        # cross moment of two BORN observables (eval_chebTT_mllct applies no
        # profile at all).  Expressing that as a degenerate profile would work
        # by accident and read as a window that is not there.
        prof = mc.get("profile") or {}

        def _w(vals, spec):
            if not spec:
                return np.ones(len(vals))
            return profile_w(vals, spec["a"], spec["b"], spec.get("c"), spec.get("d"))

        wxy = _w(X, prof.get(ox)) * _w(Y, prof.get(oy))
        rc = moments["mixed"][key]
        Cx = _cheb(_umap(np.clip(X, ax, bx), ax, bx, mpx), mc["n"])
        Cy = _cheb(_umap(np.clip(Y, ay, by), ay, by, mpy), mc["n"])
        R = float(rc["rate"])
        merr = rc.get("stat_errors" if cfg["soft_errors"] == "stat" else "errors") or {}
        for (mm, nn), v in rc["values"].items():
            F.append(Cx[:, mm] * Cy[:, nn] * wxy); mu.append(float(v) * R)
            names.append(f"{key}_T{mm}{nn}"); SIG.append(float(merr.get((mm, nn), 0.0)) * R)
        F.append(wxy.copy()); mu.append(R); names.append(f"{key}_rate"); SIG.append(_rate_err(rc))
        towers.append(dict(name=key, kind="joint", map="-", supplied=len(rc["values"]),
                           imposed=len(rc["values"]), below=[]))

    Phi = np.column_stack(F); mu = np.asarray(mu, float)
    # warm start for band variants: re-solving from the central multipliers
    # reaches the (unique) optimum of a nearby target in a fraction of a second
    sig = np.asarray(SIG, float)
    q, lam, ok = _newton(Phi, p, mu, l2=cfg["L2"], lam0=config.get("lam0"),
                         sigma=(sig if soft else None))
    if not ok or q is None:
        raise RuntimeError("MaxEnt solve did not converge to a positive-weight solution.  The usual causes: a "
                           "target the prior cannot reach (no prior events where the fixed order puts weight), "
                           "a map or window that differs from the one compiled into the fixed-order run, or "
                           "inconsistent units.  Inspect result.report after a solve with fewer orders "
                           "(config['fixed_orders']).")
    ach = (q[:, None] * Phi).sum(0)
    worst = float(np.max(np.abs(ach[1:] / np.where(np.abs(mu[1:]) > 1e-12, mu[1:], 1e-12) - 1)))
    # soft mode: the residual is by construction -sd r lam, i.e. a moment is met to within
    # its own error; report (achieved - target)/sigma per constraint (nan where sigma = 0)
    pulls = np.where(sig > 0, (ach - mu) / np.where(sig > 0, sig, 1.0), np.nan)
    effN = 1.0 / (len(q) * float((q ** 2).sum()))
    effN_retained = float((p ** 2).sum() / (q ** 2).sum())        # Kish(q) / Kish(p)
    fin = np.isfinite(pulls)
    max_pull = float(np.max(np.abs(pulls[fin]))) if fin.any() else float("nan")
    for t in towers:
        idx = [i for i, nm in enumerate(names) if nm.startswith(t["name"] + "_T") or nm == t["name"] + "_rate"]
        pt = np.abs(pulls[idx]); pt = pt[np.isfinite(pt)]
        t["max_pull"] = float(pt.max()) if len(pt) else float("nan")
    qp = q / p
    # the PRIMARY recoil (first in the config) defines the reported seam;
    # a mixed-only configuration has no single seam to report
    primary = next(iter(recoil_cfg), None)
    x_match = float(windows[primary]["x_match"]) if primary else float("nan")
    report = dict(
        moment_selection=dict(enabled=bool(do_sel), threshold=thr, chosen=dict(chosen),
                              snr={o: [float(s) for s in snr_spectra[o]] for o in snr_spectra}),
        window=(dict(windows[primary]) if primary else {}), windows=windows,
        n_constraints=len(mu),
        # band machinery: multipliers + targets always; the (large) feature
        # matrix only on request, for warm-started variants and the linear
        # response  dO = Cov_q(O, Phi) H^{-1} dmu
        lam=np.asarray(lam, float), mu=mu.copy(),
        feature_names=list(names),
        soft=dict(enabled=soft, errors=cfg["soft_errors"], sigma=sig.copy(), pulls=pulls),
        towers=towers, q_over_p_range=(float(qp.min()), float(qp.max())),
    )
    if config.get("keep_features"):
        report["Phi"] = Phi
        report["p"] = p
    chosen_all = dict(chosen)
    chosen_all.update({t["name"]: t["imposed"] for t in towers if t["kind"] == "joint"})
    return UpgradeResult(weights=np.asarray(q, float), effN=effN, closure=worst,
                         x_match=x_match, report=report, moment_snr=snr_spectra,
                         chosen_moments=chosen_all, band=None,
                         soft_pulls=(pulls if soft else None),
                         effN_retained=effN_retained, max_pull=max_pull)


def combine_moments(anchor, target, n=None):
    r"""Precision-weighted combination of two moment dicts for the SAME observable and map.

    `anchor` is the precise lower-order input (e.g. our NLO Z+jet recoil from a 40-seed
    NNLOJET run); its scale half-range per moment, `scale_errors` (and `rate_scale` for
    the window rate), is taken as the prior width on the true higher-order value.
    `target` is the higher-order Monte Carlo (e.g. an NNLO Z+jet run with large errors);
    its `stat_errors` (and `rate_stat`) are the measurement widths.  Per moment

        1/s^2 = 1/sA^2 + 1/sT^2 ,      mu = (muA/sA^2 + muT/sT^2) * s^2 ,

    and the same for the rate.  Where the target's error is much larger than the anchor's
    scale width the combination stays at the anchor; where it is much smaller it follows
    the target.  The result is a dict shaped like `target` whose values/rate are the
    posterior means and whose `stat_errors`/`rate_stat` are the posterior widths; it is
    meant to be imposed with `soft_constraints=True` (the widths then set the penalty).
    `errors`/`window_errors` (the SNR veto's total error) are the posterior width (+) the
    anchor's scale half-range.  A `_combine` block records the target's weight per moment.
    Assumes the moments are independent (no covariance is shipped by either input) and,
    for a recoil dict, that both were computed on the same window and map (checked).
    """
    rec = "window_values" in target
    if rec != ("window_values" in anchor):
        raise ValueError("anchor and target must both be recoil dicts or both Born dicts")
    if rec:
        for k in ("x_match", "x_hi", "soft_lo"):
            if not np.isclose(float(anchor.get(k, np.nan)), float(target.get(k, np.nan)), rtol=1e-9, atol=1e-9):
                raise ValueError(f"anchor and target differ in {k}: {anchor.get(k)} vs {target.get(k)}")
    vkey = "window_values" if rec else "values"; ekey = "window_errors" if rec else "errors"
    muA = np.asarray(anchor[vkey], float); muT = np.asarray(target[vkey], float)
    N = min(len(muA), len(muT)) if n is None else int(n)
    muA, muT = muA[:N], muT[:N]
    sA = np.asarray(anchor.get("scale_errors", np.zeros(N)), float)[:N]
    sT = np.asarray(target.get("stat_errors", target.get(ekey)), float)[:N]
    with np.errstate(divide="ignore", invalid="ignore"):
        pA = np.where(sA > 0, 1.0 / sA ** 2, np.inf); pT = np.where(sT > 0, 1.0 / sT ** 2, np.inf)
        both_inf = np.isinf(pA) & np.isinf(pT)
        wT = np.where(np.isinf(pA), 0.0, np.where(np.isinf(pT), 1.0, pT / (pA + pT)))
        wT = np.where(both_inf, 0.5, wT)
        s2 = np.where(np.isinf(pA) | np.isinf(pT), 0.0, 1.0 / (pA + pT))
    mu = (1.0 - wT) * muA + wT * muT; s = np.sqrt(s2)
    out = dict(target)
    out[vkey] = mu.tolist(); out["stat_errors"] = s.tolist()
    out["scale_errors"] = sA.tolist(); out[ekey] = np.hypot(s, sA).tolist()
    meta = dict(n=int(N), target_weight=wT.tolist())
    if rec:
        out["wprofile_values"] = out[vkey]
        RA = float(anchor.get("wprofile_rate", anchor["rate"])); RT = float(target.get("wprofile_rate", target["rate"]))
        rA = float(anchor.get("rate_scale", 0.0) or 0.0); rT = float(target.get("rate_stat", 0.0) or 0.0)
        if rA > 0 and rT > 0:
            pa, pt = 1.0 / rA ** 2, 1.0 / rT ** 2; wR = pt / (pa + pt); R = (1 - wR) * RA + wR * RT; rs = (pa + pt) ** -0.5
        elif rA == 0:
            wR, R, rs = 0.0, RA, 0.0
        else:
            wR, R, rs = 1.0, RT, 0.0
        out["rate"] = out["wprofile_rate"] = float(R); out["rate_stat"] = float(rs); out["rate_scale"] = rA
        meta.update(rate_target_weight=float(wR), rate_anchor=RA, rate_target=RT)
    out["_combine"] = meta
    return out
