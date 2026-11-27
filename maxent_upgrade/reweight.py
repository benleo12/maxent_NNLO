"""Positive event weights from fixed-order moment constraints.

Read upgrade() for the calculation: validate, build the measurement functions,
select moments, solve the convex dual, and report the achieved constraints.
"""
import warnings

import numpy as np

from .basis import cheb as _cheb, umap as _umap
from .constraints import chebyshev_moment, profile_w, resolved_order, _window_fn, _window_edges
from .result import UpgradeResult
from .solver import _newton_maxent as _newton
from .support import verify_support

DEFAULTS = dict(
    weight_key="weight",
    moment_selection=True,
    snr_threshold=1.0,
    L2=1e-4,
    fixed_orders=None,
    soft_constraints=False,
    soft_errors="stat",
)

# config keys upgrade() reads, and the legacy keys only upgrade_from_histograms() reads
_UPGRADE_KEYS = {"born", "recoil", "mixed", "followers", "weight_key", "moment_selection",
                 "snr_threshold", "L2", "fixed_orders", "lam0", "keep_features",
                 "soft_constraints", "soft_errors"}
_HIST_ONLY_KEYS = {"band", "RELMAX", "DELTA", "WSIG", "CLIP", "MINSUP", "RSUP", "born_N",
                   "recoil_N", "snr_max_order", "snr_n_stat", "snr_seed"}
# how a recoil tower meets the prior across its window (config['recoil'][obs]['transition'])
_TRANSITIONS = ("blend", "windowed")
_DEFAULT_TRANSITION = "blend"


def _transition(rcfg, obs):
    tr = rcfg.get("transition", _DEFAULT_TRANSITION)
    if tr not in _TRANSITIONS:
        raise ValueError(
            f"config['recoil']['{obs}']['transition'] must be 'blend' or 'windowed', got {tr!r}")
    return tr


def _moment_snr(difference, error):
    """Absolute moment shift in units of the quoted error, including zero-error inputs."""
    difference, error = np.broadcast_arrays(np.abs(difference), error)
    # All Chebyshev averages lie in [-1, 1]. Allow for roundoff on this scale.
    zero_error = np.where(difference <= 32 * np.finfo(float).eps, 0., np.inf)
    return np.divide(difference, error, out=zero_error, where=error > 0)


def _check_event_columns(events, need, wkey):
    """Validate the prior before a fit or a new fixed-order integration."""
    need = list(dict.fromkeys(need))
    missing = [k for k in need + [wkey] if k not in events]
    if missing:
        raise KeyError(f"events lacks {missing}. This config needs one array per constrained observable "
                       f"{need} plus the prior weights '{wkey}'; events has {sorted(events)}")
    for key in [wkey] + need:
        values = np.asarray(events[key])
        if values.ndim != 1:
            raise ValueError(
                f"events['{key}'] must be a one-dimensional array, got shape {values.shape}")
        if values.size == 0:
            raise ValueError(
                f"events['{key}'] is empty: provide events inside the target's fiducial selection")
    n = len(np.asarray(events[wkey]))
    bad = {k: len(np.asarray(events[k])) for k in need if len(np.asarray(events[k])) != n}
    if bad:
        raise ValueError(f"events arrays differ in length: '{wkey}' has {n}, but {bad}")
    for k in need + [wkey]:
        nf = int((~np.isfinite(np.asarray(events[k], float))).sum())
        if nf:
            raise ValueError(
                f"events['{k}'] has {nf} non-finite entries; remove those events first")
    w = np.asarray(events[wkey], float)
    if not np.all(w > 0):
        raise ValueError(f"prior weights must be strictly positive: {int((w <= 0).sum())} of {len(w)} are not "
                         f"(provide a positive-weight prior; signed fixed-order weights belong in the moment calculation)")
    if not np.isfinite(w.sum()):
        raise ValueError(
            "the sum of prior weights is not finite: rescale the input weights before reweighting")


def _check_inputs(events, moments, config, wkey):
    """Fail early, with a message that says what to fix, on inconsistent inputs."""
    for k in config:
        if k in _HIST_ONLY_KEYS:
            warnings.warn(f"config['{k}'] is read only by the legacy upgrade_from_histograms(); "
                          f"upgrade() ignores it", stacklevel=3)
        elif k not in _UPGRADE_KEYS:
            warnings.warn(f"config['{k}'] is not a key upgrade() knows; it is ignored "
                          f"(known: {sorted(_UPGRADE_KEYS)})", stacklevel=3)
    born_cfg = config.get("born", {}) or {}
    recoil_cfg = config.get("recoil", {}) or {}
    mixed_cfg = config.get("mixed", {}) or {}
    if not recoil_cfg and not mixed_cfg and not born_cfg:
        raise ValueError(
            "config declares no constraint: give at least one of 'born', 'recoil', 'mixed'")
    for o, rc in recoil_cfg.items():
        _transition(rc, o)
    # Joint moments are imposed in the windowed form only: a blended tower mixes prior and
    # fixed order through its own window, while a joint moment is pure fixed order, so
    # blended towers are in general inconsistent with fixed-order joint moments.  Every tower
    # a joint block couples must therefore be windowed too.
    for key, mc in mixed_cfg.items():
        if not isinstance(mc.get("include_rate", True), (bool, np.bool_)):
            raise ValueError(f"config['mixed'][{key!r}]['include_rate'] must be boolean")
        if mc.get("transition", "windowed") != "windowed":
            raise ValueError(f"config['mixed']['{key}']['transition'] = {mc['transition']!r}: joint blocks are "
                             f"imposed in the windowed form only (blended towers are in general inconsistent with fixed-order joint moments)")
        for o in mc["observables"]:
            if o in recoil_cfg and _transition(recoil_cfg[o], o) == "blend":
                raise ValueError(
                    f"config['mixed']['{key}'] couples the recoil tower '{o}', whose transition is 'blend'. "
                    f"Joint moments are imposed in the windowed form, and blended towers are in general "
                    f"inconsistent with fixed-order joint moments: set config['recoil']['{o}']['transition'] = "
                    f"'windowed' for every tower the block couples (load_moments does this for a table's joint "
                    f"blocks)")
    need = list(born_cfg) + list(recoil_cfg) + \
        [o for m in mixed_cfg.values() for o in m["observables"]]
    _check_event_columns(events, need, wkey)
    for role, keys in (("born", born_cfg), ("recoil", recoil_cfg), ("mixed", mixed_cfg)):
        for k in keys:
            if k not in (moments.get(role) or {}):
                raise KeyError(f"config['{role}'] names '{k}', but moments['{role}'] has no entry for it "
                               f"(has {sorted(moments.get(role) or {})})")
    for o, bc in born_cfg.items():
        a, b = bc["range"]
        x = np.asarray(events[o], float)
        out = float(np.mean((x < a) | (x > b)))
        if out > 0.05:
            warnings.warn(f"{100 * out:.1f}% of the prior's '{o}' lies outside the tower range [{a:g}, {b:g}]; "
                          f"those events are clipped to the edge.  Same fiducial cuts and units on both sides?",
                          stacklevel=3)
    for o, rc in recoil_cfg.items():
        mrc = moments["recoil"][o]
        x = np.asarray(events[o], float)
        prof = rc.get("profile")
        if prof and prof.get("a") is not None and not np.isclose(float(prof["a"]), float(mrc["x_match"]),
                                                                 rtol=1e-9, atol=0.0):
            warnings.warn(f"config['recoil']['{o}']['profile']['a'] = {float(prof['a']):g}, but the moments were "
                          f"booked with a window starting at x_match = {float(mrc['x_match']):g}.  The window is "
                          f"part of the target: use the profile the moments were computed with", stacklevel=3)
        a, b, c, d = _window_edges(prof, mrc["x_match"], mrc.get(
            "x_hi", rc.get("range", (None, np.inf))[1]))
        win = profile_w(x, a, b, c, d) > 0
        nin = int(win.sum())
        if nin == 0:
            raise ValueError(f"no prior event lies inside the '{o}' window (from {float(mrc['x_match']):g}); "
                             f"check the units (GeV) and the fiducial selection of events['{o}']")
        if nin < 100:
            warnings.warn(f"only {nin} prior events lie inside the '{o}' window; the upgrade there is "
                          f"Monte Carlo noise times a tilt", stacklevel=3)
        hi = float(mrc.get("x_hi", rc.get("range", (None, np.inf))[1]))
        above = float(np.mean(x > hi))
        if above > 0.05:
            warnings.warn(f"{100 * above:.1f}% of the prior's '{o}' lies above the tower range ({hi:g}); "
                          f"check the units (GeV)", stacklevel=3)


def upgrade(events, moments, config, *, check_support=False):
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
                               'stat_errors': [..], 'scale_errors': [..],
                               'rate_stat': .., 'rate_scale': ..}},
              'mixed':  {key: {'values': {(m, n): mu_mn}, 'errors': {..}, 'rate': r}},  # optional
            }

        Born: values[n-1] = sum_j w_j T_n(u(x_j)) / sum_j w_j over the fixed-order points.
        Recoil: the windowed moments <w T_n(u)>/<w> with u the log map on [soft_lo, x_hi]
        and w the window of `config['recoil'][obs]['profile']` (or the hard window
        [x_match, x_hi) without a profile), and the window rate r = <w>/<1>.  `errors` are
        the total fixed-order uncertainties (Monte Carlo scatter combined with the scale
        envelope); they decide which orders are resolved.  `load_moments(process)` returns
        this dict for the released processes; `compute_fo_moments` builds it from FO events.
    config : dict
        Roles, maps and windows, plus optional knobs::

            config = {
              'born':   {'mll':   {'range': (66., 116.), 'map': 'bw'},   # 'lin', 'log', 'bw'
                         'y_abs': {'range': (0., 2.4),   'map': 'lin'}},
              'recoil': {'pT_ll': {'range': (30., 2500.), 'map': 'log', 'soft_lo': 30.,
                                   'profile': {'a': 30., 'b': 60., 'c': 500.}}},
            }

        A recoil tower takes `transition` = 'blend' (the default) or 'windowed':
          * 'blend': the target of order n is the mixture
                mu_n = c <(1 - w) T_n>_p + r <T_n>_{FO, w},   c = (1 - r)/(1 - <w>_p),
            the prior's own moment weighted by 1 - w and the fixed-order one by w, and the
            measurement functions are the bare T_n(u(v)), clipped at the window's lower edge
            a (and at d for a smooth upper edge c -> d), so this tower contributes a constant
            tilt below a. A hard upper cut c instead zeroes the
            features above c and fixes the prior's share there, c <1(v >= c)>_p.
          * 'windowed': the measurement functions are w(v) T_n(u(v)) and w(v), with targets
            r <T_n>_{FO, w} and r; this tower contributes no tilt where w = 0.
        Other correlated constraints can still change the marginal distribution in these regions.
        Joint blocks are always imposed in the windowed form, and every recoil tower a block
        couples must be 'windowed' (a ValueError says so otherwise): blended towers are in
        general inconsistent with fixed-order joint moments.

        Knobs (defaults in DEFAULTS): moment_selection=True (impose each tower up to its
        highest order with SNR > snr_threshold=1.0), L2=1e-4 (ridge of the dual), fixed_orders
        ({obs: N}: impose exactly N orders, e.g. for uncertainty re-solves), lam0 (warm start from a
        previous result.report['lam']), keep_features=False (return Phi and p in the report),
        soft_constraints=False and soft_errors='stat' (Gaussian soft constraints),
        weight_key='weight'.  Unknown keys raise a warning.

    check_support : bool
        Check imposed hard targets against the input events and retain per-feature
        means, errors and residuals. The portable fit_targets workflow enables this.

    Returns
    -------
    UpgradeResult (see its docstring): weights, effN, effN_retained, closure, max_pull,
    moment_snr, chosen_moments, x_match, report; result.describe() prints a summary table.
    """
    cfg = {**DEFAULTS, **config}
    wkey = cfg["weight_key"]
    born_cfg = config.get("born", {}) or {}
    recoil_cfg = config.get("recoil", {}) or {}
    if wkey not in events:
        raise KeyError(f"events lacks the prior weights '{wkey}' (events has {sorted(events)})")
    w = np.asarray(events[wkey], float)
    _check_inputs(events, moments, config, wkey)
    p = w / w.sum()
    thr = cfg["snr_threshold"]
    do_sel = cfg["moment_selection"]

    F = [np.ones(len(w))]
    mu = [1.0]
    names = ["norm"]
    measurements = [(None, None, "normalization", 0.0)]
    correlated_errors = {}
    SIG = [0.0]      # per-constraint target error (feature units); used only in soft mode
    # indices of the window-share constraints (a window rate, the share above a cut)
    SHARES = []
    snr_spectra = {}
    prefit = []
    chosen = {}
    towers = []     # towers: one row per tower for describe()
    soft = bool(cfg["soft_constraints"])

    def record_prefit(name, degree, prior, target, error, used, rule):
        # Conditional window moments use their own denominator, as in the input table.
        snr = None
        if prior is not None and error is not None:
            value = float(_moment_snr(target - prior, error))
            snr = value if np.isfinite(value) else None
        prefit.append(dict(observable=name, degree=degree, prior=prior, target=float(target),
                           error=error, snr=snr, retained=bool(used), rule=rule))

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

    def rate_error(d):
        if "rate_stat" in d or "rate_scale" in d:
            return float(np.hypot(d.get("rate_stat", 0.) or 0., d.get("rate_scale", 0.) or 0.))
        return None

    def effective_error(coefficient, error, rate_coefficient, rate_uncertainty):
        if (coefficient and error is None) or (rate_coefficient and rate_uncertainty is None):
            return None
        return float(np.hypot(coefficient * (error or 0.), rate_coefficient * (rate_uncertainty or 0.)))

    def grouped_error(info, rate, moment, prior_part):
        # Effective target = (1 - r) * prior_part + r * moment. The shared run
        # covariance supplies the cross term that separate marginal errors lack.
        dr = moment - prior_part
        variance = (rate * info["moment_stat"])**2 + (dr * info["rate_stat"])**2 + 2 * rate * dr * info["cov"]
        stat = float(np.sqrt(max(0., variance)))
        rates = np.asarray(info["scale_rates"])
        varied = (1 - rates) * prior_part + rates * np.asarray(info["scale_moments"])
        total = float(np.hypot(stat, np.ptp(varied) / 2))
        SIG[-1] = stat if cfg["soft_errors"] == "stat" else total
        correlated_errors[len(SIG) - 1] = stat
        return total

    # ---- ONE rule for every tower -----------------------------------------
    # A Born tower imposes the fixed-order moments over the full fiducial range.  A recoil
    # tower imposes them through its window w: blended with the prior across the window's
    # rise (transition 'blend', the default) or with the features masked by w ('windowed').
    def _impose(obs, lo_map, hi_map, mp, vals, errs, floor, sig=None, wprof=None, rate=1.0,
                rate_sig=0.0, transition=None, lo_feat=None, hi_feat=None, hi_cut=None, degrees=None, rate_uncertainty=None, grouped=None):
        XL = np.asarray(events[obs], float)
        count = len(vals)
        errs = np.broadcast_to(errs, (count,)) if len(errs) else errs
        degrees = np.arange(1, count + 1, dtype=int) if degrees is None else np.asarray(degrees)
        if (degrees.shape != (count,) or degrees.dtype.kind not in "iu"
                or (degrees < 1).any() or (np.diff(degrees.astype(int)) <= 0).any()):
            raise ValueError(f"{obs}: degrees must be increasing positive integers, one per target")
        highest = int(degrees[-1]) if count else 0
        sigv = np.zeros(count) if sig is None or not len(sig) else np.asarray(sig, float)[:count]
        if len(sigv) < count:
            sigv = np.concatenate([sigv, np.zeros(count - len(sigv))])
        # The prior's moment in the resolution criterion is the same quantity as the booked
        # target: for a window tower the window-weighted <w T_n>/<w>, for a Born tower the
        # plain average over the full range.
        if wprof is not None:
            wpr = np.asarray(wprof, float)
            win = wpr > 0
            mu_prior = (chebyshev_moment(XL[win], (w * wpr)[win], highest, lo_map, hi_map, mp)
                        if win.any() else np.zeros(highest))
        else:
            mu_prior = chebyshev_moment(XL, w, highest, lo_map, hi_map, mp)
        mu_prior = mu_prior[degrees - 1]
        snr = (_moment_snr(vals - mu_prior, errs)
               if len(errs) else np.full(count, np.inf))
        snr_spectra[obs] = snr
        n_imposed = max(resolved_order(snr, thr) if do_sel else count, floor)
        # a band variant (scale choice, bootstrap replica) must impose the SAME
        # constraint set as the central solve: near the SNR threshold the rule can
        # flip an order in or out between variants, which would change the number
        # of multipliers and make the variant a different problem, not a shifted one
        fixed = (cfg.get("fixed_orders") or {}).get(obs)
        if fixed is not None:
            n_imposed = min(max(int(fixed), floor), count)
        if check_support:
            rule = "specified" if fixed is not None or not do_sel else "individual spectrum"
            for i, degree in enumerate(degrees):
                prior = float(mu_prior[i]) if wprof is None or win.any() else None
                error = float(errs[i]) if len(errs) else None
                record_prefit(obs, int(degree), prior, vals[i], error, i < n_imposed, rule)
        chosen[obs] = n_imposed
        towers.append(dict(name=obs, kind="born" if wprof is None else transition,
                           map=mp if isinstance(mp, str) else mp[0],
                           supplied=int(count), imposed=int(n_imposed), supplied_degrees=degrees.tolist(),
                           fitted_degrees=degrees[:n_imposed].tolist(),
                           below=[(int(n), float(s)) for n, s in zip(degrees[:n_imposed], snr[:n_imposed]) if s <= thr]))
        # Under the blend the features are clipped at the window's lower edge (and at d for a
        # smooth upper edge): every event below it carries the same T_n, so the tilt is flat
        # there and the shower's shape is kept exactly.  The booked moments are unaffected,
        # because w vanishes below the edge, where the clipping acts.
        lo_c = lo_map if lo_feat is None else lo_feat
        hi_c = hi_map if hi_feat is None else hi_feat
        u = _umap(np.clip(XL, lo_c, hi_c), lo_map, hi_map, mp)
        C = _cheb(u, int(degrees[n_imposed - 1]) if n_imposed else 0)
        if wprof is None:
            # -------- Born tower: the fixed-order moment over the full range --------
            for i, n in enumerate(degrees[:n_imposed]):
                F.append(C[:, n])
                mu.append(float(vals[i]))
                names.append(f"{obs}_T{n}")
                SIG.append(float(sigv[i]))
                measurements.append((obs, int(n), "born", float(errs[i]) if len(errs) else None))
        elif transition == "blend":
            # -------- blend: the target density on this axis is the smooth mixture --------
            #     b(x) = (1 - w(x)) * cs * p(x) + w(x) * f(x),   cs = (1 - R) / (1 - R_p),
            # with p the prior, f the fixed-order density (normalised to the fiducial cross
            # section, so its window share is R) and R_p = <w>_p the prior's window share.
            # The prior part is rescaled by cs so that b integrates to one: the prior keeps its
            # shape and carries 1 - R, the fixed order keeps its shape and its rate where
            # w = 1, and the two are mixed through w across the rise of the window instead of
            # being joined at the seam.  Its Chebyshev moments are
            #     mu_n = cs <(1 - w) T_n>_p + R mu_n^{FO,w},
            # with mu_n^{FO,w} the w-weighted fixed-order moments already booked (`vals`), so
            # no new fixed-order input is needed.  The features are T_n(u) themselves below a
            # hard upper cut x_c (zero above it, where one extra constraint fixes the share to
            # its blend value cs <1(x >= x_c)>_p); a smooth fall-off [c, d] needs no extra
            # constraint, since the features are clipped at d.  The window share itself is not
            # imposed separately: the moments carry it.
            R = float(rate)
            wprof = np.asarray(wprof, float)
            inside = np.ones(len(XL)) if hi_cut is None else (XL < float(hi_cut)).astype(float)
            R_p = float((p * wprof).sum())
            if R_p >= 1.0:
                raise ValueError(f"the '{obs}' window holds every prior event (<w>_p = 1), so the blend has no "
                                 f"prior part to keep: use transition='windowed' for this tower")
            cs = (1.0 - R) / (1.0 - R_p)
            for i, n in enumerate(degrees[:n_imposed]):
                # <(1-w) T_n>_p below x_c
                prior_part = float((p * (1.0 - wprof) * inside * C[:, n]).sum())
                F.append(C[:, n] * inside)
                mu.append(cs * prior_part + R * float(vals[i]))
                names.append(f"{obs}_T{n}")
                # d mu_n / dR = mu_n^{FO,w} - <(1-w) T_n>_p / (1 - R_p)
                dR = float(vals[i]) - prior_part / (1.0 - R_p)
                SIG.append(float(np.hypot(float(sigv[i]) * R, dR * float(rate_sig))))
                error = effective_error(R, errs[i] if len(errs) else None, dR, rate_uncertainty)
                if grouped and n in grouped:
                    error = grouped_error(grouped[n], R, vals[i], prior_part / (1.0 - R_p))
                measurements.append((obs, int(n), "blend", error))
            above = 1.0 - inside
            if above.any():
                P_above = float((p * above).sum())
                SHARES.append(len(names))
                F.append(above)
                mu.append(cs * P_above)
                names.append(f"{obs}_above")
                SIG.append(float(P_above / (1.0 - R_p) * float(rate_sig)))
                error = effective_error(0., None, P_above / (1.0 - R_p), rate_uncertainty)
                measurements.append((obs, None, "tail fraction", error))
        else:
            # -------- windowed: features masked by the window --------
            # feature C[:,n]*w with target <T_n>_FO(w-weighted) * R, plus an explicit rate
            # constraint sum q*w = R.  Where w = 0 (below the window, above its cut) every
            # feature vanishes, so those events are not tilted by this tower and the prior's
            # shape there is kept exactly; across the rise the tilt grows smoothly with w (C^2)
            # wherever prior and fixed order disagree there.
            R = float(rate)
            wprof = np.asarray(wprof, float)
            for i, n in enumerate(degrees[:n_imposed]):
                F.append(C[:, n] * wprof)
                mu.append(float(vals[i]) * R)
                names.append(f"{obs}_T{n}")
                SIG.append(float(np.hypot(float(sigv[i]) * R,
                           float(vals[i]) * float(rate_sig))))
                error = effective_error(R, errs[i] if len(errs) else None, vals[i], rate_uncertainty)
                if grouped and n in grouped:
                    error = grouped_error(grouped[n], R, vals[i], 0.)
                measurements.append((obs, int(n), "windowed", error))
            SHARES.append(len(names))
            F.append(wprof.copy())
            mu.append(R)
            names.append(f"{obs}_rate")
            SIG.append(float(rate_sig))
            measurements.append((obs, None, "window fraction", rate_uncertainty))

    # Born observables: validity is the full fiducial range.
    for o in born_cfg:
        a, b = born_cfg[o]["range"]
        mp = born_cfg[o].get("map", "lin")
        _impose(o, a, b, mp,
                np.asarray(moments["born"][o]["values"], float),
                np.asarray(moments["born"][o].get("errors", []), float),
                floor=0, sig=_soft_err(moments["born"][o], "errors"), degrees=moments["born"][o].get("degrees"))
    # Recoil observables, each through its own window (there may be more than one, e.g. the
    # two jet spectra of Z+jet, or a recoil and a decay-angle window).  Each carries its own
    # profile, rate and transition; their features add to the same feature matrix.
    windows = {}
    for recoil_obs, rcfg in recoil_cfg.items():
        rc = moments["recoil"][recoil_obs]
        XL = np.asarray(events[recoil_obs], float)
        xm = float(rc["x_match"])
        XHI = float(rc["x_hi"]) if "x_hi" in rc else float(rcfg["range"][1])   # the booked interval
        soft_lo = float(rc.get("soft_lo", rcfg.get("soft_lo", max(rcfg["range"][0], 1e-6))))
        a, b, c, d = _window_edges(rcfg.get("profile"), xm, XHI)
        trans = _transition(rcfg, recoil_obs)
        windows[recoil_obs] = dict(x_match=xm, x_hi=XHI, rate=float(rc["rate"]),
                                   transition=trans, profile=dict(a=a, b=b, c=c, d=d))
        wprof = profile_w(XL, a, b, c, d)
        # `vals`/`rate` are the w-weighted FO moments and the w-rate (wprofile_* where a
        # producer distinguishes them from a hard window's values)
        vals = np.asarray(rc.get("wprofile_values", rc["window_values"]), float)
        R = float(rc.get("wprofile_rate", rc["rate"]))
        smooth_top = c is not None and d is not None and d > c
        blend = trans == "blend"
        _impose(recoil_obs, soft_lo, XHI, rcfg.get("map", "log"), vals,
                np.asarray(rc.get("window_errors", []), float), floor=1,
                sig=_soft_err(rc, "window_errors"), wprof=wprof, rate=R, rate_sig=_rate_err(rc),
                transition=trans, degrees=rc.get("degrees"), rate_uncertainty=rate_error(rc), grouped=rc.get("grouped_errors"),
                lo_feat=(max(soft_lo, a) if blend else None),
                hi_feat=(min(XHI, d) if (blend and smooth_top) else None),
                hi_cut=(None if (not blend or c is None or smooth_top) else c))

    # ---- JOINT (two-observable) constraints, always in the windowed form ---------------
    # Separate moments can change the joint distribution through correlations already
    # present in the prior. A product moment adds an explicit joint constraint. Its feature is
    #     T_m(u(x)) T_n(v(y)) w(x) w(y)
    # with target <T_m T_n>_w * R_xy, plus the joint rate constraint sum q w w.  Both
    # profiles multiply, so the feature vanishes unless BOTH observables are inside their
    # windows.  A missing or None profile means no window (w = 1): the feature is the bare
    # product, which is what NNLOJET books for two Born observables, and there is no rate.
    for key, mc in (config.get("mixed") or {}).items():
        ox, oy = mc["observables"]
        ax, bx = mc["range"][ox]
        ay, by = mc["range"][oy]
        mpx = mc.get("map", {}).get(ox, "log")
        mpy = mc.get("map", {}).get(oy, "log")
        X = np.asarray(events[ox], float)
        Y = np.asarray(events[oy], float)
        prof = mc.get("profile") or {}
        wxy = _window_fn(X, prof.get(ox)) * _window_fn(Y, prof.get(oy))
        rc = moments["mixed"][key]
        Cx = _cheb(_umap(np.clip(X, ax, bx), ax, bx, mpx), mc["n"])
        Cy = _cheb(_umap(np.clip(Y, ay, by), ay, by, mpy), mc["n"])
        R = float(rc["rate"])
        merr = rc.get("stat_errors" if cfg["soft_errors"] == "stat" else "errors") or {}
        prior_window = wxy @ p if check_support else None
        for (mm, nn), v in rc["values"].items():
            F.append(Cx[:, mm] * Cy[:, nn] * wxy)
            mu.append(float(v) * R)
            names.append(f"{key}_T{mm}{nn}")
            SIG.append(float(merr.get((mm, nn), 0.0)) * R)
            total = (rc.get("errors") or {}).get((mm, nn))
            if check_support:
                prior = float(p @ F[-1] / prior_window) if prior_window > 0 else None
                record_prefit(key, [mm, nn], prior, v, total, True, "supplied joint moment")
            fraction_error = rate_error(rc) if prof.get(ox) or prof.get(oy) else 0.
            kind = "windowed joint" if prof.get(ox) or prof.get(oy) else "joint"
            error = effective_error(R, total, v, fraction_error)
            if (mm, nn) in rc.get("grouped_errors", {}):
                error = grouped_error(rc["grouped_errors"][(mm, nn)], R, v, 0.)
            measurements.append((key, [mm, nn], kind, error))
        if (prof.get(ox) or prof.get(oy)) and mc.get("include_rate", True):
            SHARES.append(len(names))
            F.append(wxy.copy())
            mu.append(R)
            names.append(f"{key}_rate")
            SIG.append(_rate_err(rc))
            measurements.append((key, None, "window fraction", rate_error(rc)))
        towers.append(dict(name=key, kind="joint", map="-", supplied=len(rc["values"]),
                           imposed=len(rc["values"]), fitted_degrees=[list(pair) for pair in rc["values"]], below=[]))

    Phi = np.column_stack(F)
    mu = np.asarray(mu, float)
    F.clear()                    # the columns live on in Phi; free the copies (large priors)
    # warm start for band variants: re-solving from the central multipliers
    # reaches the (unique) optimum of a nearby target in a fraction of a second
    sig = np.asarray(SIG, float)
    if check_support:
        prior_means = p @ Phi
        prior_means[0] = p.sum()
        hard = sig == 0 if soft else np.ones_like(sig, dtype=bool)
        hard[0] = False  # all trial distributions are normalized by construction
        columns = slice(1, None) if np.all(hard[1:]) else np.flatnonzero(hard)
        support = verify_support(Phi[:, columns], mu[columns], prior_means[columns], np.asarray(names)[columns])
        if support.endswith("unresolved"):
            warnings.warn("Hard-target convex-hull membership is unresolved. Inspect the achieved moments.",
                          stacklevel=2)
    q, lam, ok = _newton(Phi, p, mu, l2=cfg["L2"], lam0=config.get("lam0"),
                         sigma=(sig if soft else None))
    if not ok or q is None:
        raise RuntimeError("MaxEnt solve did not converge to a positive-weight solution.  The usual causes: a "
                           "target the prior cannot reach (no prior events where the fixed order puts weight), "
                           "a map or window that differs from the one compiled into the fixed-order run, or "
                           "inconsistent units.  Inspect result.report after a solve with fewer orders "
                           "(config['fixed_orders']).")
    if check_support:
        ach = q @ Phi
        ach[0] = q.sum()
    else:
        ach = (q[:, None] * Phi).sum(0)
    # A zero target has no relative scale. Report numerical zero as closed and
    # retain any resolved absolute residual separately instead of inventing a
    # denominator (which made an exactly achieved zero look like 100% error).
    absolute_residual = ach - mu
    rel = np.divide(np.abs(absolute_residual), np.abs(mu),
                    out=np.full_like(mu, np.nan), where=mu != 0)
    rel[(mu == 0) & (np.abs(absolute_residual) <= 1e-12)] = 0
    # closure and pulls are quoted for the moments; the window shares (the rate of a windowed
    # tower or joint block, the share above a blended tower's hard cut) are reported apart
    is_mom = np.ones(len(mu), bool)
    is_mom[0] = False
    is_mom[SHARES] = False
    worst = float(rel[is_mom].max()) if is_mom.any() else 0.0
    # soft mode: the residual is by construction -sd r lam, i.e. a moment is met to within
    # its own error; report (achieved - target)/sigma per constraint (nan where sigma = 0)
    pulls = np.where(sig > 0, (ach - mu) / np.where(sig > 0, sig, 1.0), np.nan)
    effN = 1.0 / (len(q) * float((q ** 2).sum()))
    effN_retained = float((p ** 2).sum() / (q ** 2).sum())        # Kish(q) / Kish(p)
    fin = np.isfinite(pulls) & is_mom
    max_pull = float(np.max(np.abs(pulls[fin]))) if fin.any() else float("nan")
    shares = {names[i]: dict(closure=float(rel[i]), pull=float(pulls[i])) for i in SHARES}
    for t in towers:
        idx = [i for i, nm in enumerate(names) if is_mom[i] and nm.startswith(t["name"] + "_T")]
        pt = np.abs(pulls[idx])
        pt = pt[np.isfinite(pt)]
        t["max_pull"] = float(pt.max()) if len(pt) else float("nan")
    qp = q / p
    # the PRIMARY recoil (first in the config) defines the reported seam;
    # a mixed-only configuration has no single seam to report
    primary = next(iter(recoil_cfg), None)
    x_match = float(windows[primary]["x_match"]) if primary else float("nan")
    report = dict(
        fitted_degrees={t["name"]: t["fitted_degrees"] for t in towers},
        moment_selection=dict(enabled=bool(do_sel), threshold=thr, chosen=dict(chosen),
                              snr={o: [float(s) for s in snr_spectra[o]] for o in snr_spectra}),
        window=(dict(windows[primary]) if primary else {}), windows=windows,
        n_constraints=len(mu),
        # band machinery: multipliers + targets always; the (large) feature
        # matrix only on request, for warm-started variants and the linear
        # response  dO = Cov_q(O, Phi) H^{-1} dmu
        lam=np.asarray(lam, float), mu=mu.copy(),
        feature_names=list(names),
        zero_target_residuals={names[i]: float(absolute_residual[i]) for i in np.flatnonzero(mu == 0)},
        soft=dict(enabled=soft, errors=cfg["soft_errors"], sigma=sig.copy(), pulls=pulls),
        towers=towers, shares=shares, q_over_p_range=(float(qp.min()), float(qp.max())),
    )
    if check_support:
        report["support"] = support
        report["prefit"] = prefit
        report["constraints"] = [
            dict(name=name, observable=observable, degree=degree, kind=kind,
                 mode="soft" if soft and sig[i] > 0 else "hard", prior=float(prior_means[i]),
                 target=float(mu[i]), achieved=float(ach[i]), residual=float(absolute_residual[i]),
                 input_error=error)
            for i, (name, (observable, degree, kind, error)) in enumerate(zip(names, measurements))
        ]
        for i, statistical in correlated_errors.items():
            report["constraints"][i].update(integration_error=statistical,
                                            input_error_model="moment-rate covariance and coherent scales")
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
