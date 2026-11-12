"""Measurement functions and fixed-order moments for user-defined observables.

The same maps and windows must be used on the prior and in the fixed-order calculation.
"""
import warnings

import numpy as np

from .basis import cheb as _cheb, umap as _umap


def resolved_order(snr, threshold=1.0) -> int:
    """Largest moment order n with SNR_n > threshold (0 if none)."""
    snr = np.asarray(snr, float)
    hits = [n for n in range(1, len(snr) + 1) if snr[n - 1] > threshold]
    return max(hits, default=0)


def chebyshev_moment(x, w, n_max, a, b, mp):
    r"""Event-level Chebyshev moments <T_1..T_n_max> of observable x under weights w.

        mu_n = sum_i w_i T_n(u(x_i)) / sum_i w_i ,   u = umap(x, a, b, mp)

    This is the weighted sum a fixed-order code performs when it books the observable
    T_n(u(x)); there is no binning.  `mp` is 'lin', 'log', 'bw' or ('bw', M, Gamma);
    values outside [a, b] are clipped to the edge, as in the NNLOJET patch.
    """
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    u = _umap(np.clip(x, a, b), a, b, mp)
    C = _cheb(u, n_max)
    s = w.sum()
    return np.array([(w * C[:, n]).sum() / s for n in range(1, n_max + 1)])


def alpha_s(mu, mz=91.1876, az=0.118, nf=5):
    """Two-loop running coupling, normalised at m_Z."""
    mu = np.asarray(mu, float)
    b0 = (33 - 2 * nf) / (12 * np.pi)
    b1 = (153 - 19 * nf) / (24 * np.pi ** 2)
    L = np.log(np.maximum(mu, 1.0) ** 2 / mz ** 2)
    x = np.maximum(1 + az * b0 * L, 1e-9)
    return az / (x + az * (b1 / b0) * np.log(x))


def matching_scale(Q, thr=0.2, lo=3.0, hi=None, n=4000, color=1.0):
    r"""The matching scale v_m for a process with hard scale Q, from power counting:

        v_m = min { pT :  alpha_s(pT) (C_i/C_F) ln^2(Q/pT)  <  thr }      (thr = 0.2)

    Below it unresummed Sudakov logarithms spoil the fixed-order series and the shower
    is kept; above it the fixed order is imposed.  `color` is C_i/C_F, the color factor of
    the incoming partons relative to quarks: 1 for quark-initiated processes, C_A/C_F = 9/4
    for gluon fusion.  With color = 1: 28.0 GeV for Q = m_Z, 25.0 GeV for Q = m_W, 27.7 GeV
    for Q = 90 GeV (diphoton), 37.3 GeV for Q = m_H.  With color = 9/4 and Q = m_H: 54.3 GeV,
    the released Higgs window.  Returns the scale in GeV.
    """
    Q = float(Q)
    hi = 0.9 * Q if hi is None else float(hi)
    pt = np.geomspace(lo, hi, int(n))
    v = float(color) * alpha_s(pt) * np.log(Q / np.minimum(pt, 0.99 * Q)) ** 2
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
              f"({100 * (x_match - want) / want:+.0f}%).  The window is compiled into the fixed-order "
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


def _window_fn(x, spec):
    """The window of one observable of a joint block: 1 without a profile, else profile_w."""
    if not spec:
        return np.ones(len(x))
    return profile_w(x, float(spec["a"]), float(spec["b"]),
                     None if spec.get("c") is None else float(spec["c"]),
                     None if spec.get("d") is None else float(spec["d"]))


def _window_edges(prof, x_match, x_hi):
    """(a, b, c, d) of a recoil window; without a profile, the hard window [x_match, x_hi)."""
    if not prof:
        return float(x_match), float(x_match), float(x_hi), None
    a = float(prof.get("a", x_match))
    b = float(prof.get("b", a))
    c = prof.get("c", x_hi)
    d = prof.get("d", None)
    return a, b, (None if c is None else float(c)), (None if d is None else float(d))


def compute_fo_moments(fo_events, config, x_match=None, x_hi=None, n_max=None,
                       weight_key="weight", scale_key="weight_scales"):
    r"""Build the `moments` argument of `upgrade` from fixed-order EVENTS held in Python.

    Use this only if you have the fixed-order phase-space points in memory; otherwise
    accumulate the same weighted sums inside your fixed-order code (MOMENT_CONSTRUCTION.md)
    and fill the dict by hand.  `fo_events` is a dict of 1D arrays: every constrained
    observable, the weights under `weight_key`, and optionally a (n_scale, n_event) array
    under `scale_key` with the weight of each scale choice (row 0 = central).

    * Born towers: <T_n(u(v))> over the full declared range.
    * Recoil towers: the window of `config['recoil'][obs]['profile']` (a, b, c[, d]) when one
      is given, otherwise the hard window [x_match, x_hi).  They get the windowed moments
      <w T_n>/<w> and the window rate <w>/<1>; upgrade() forms the blended (or windowed)
      targets from them, so the same numbers serve either transition.
    * Joint blocks of `config['mixed']`: <w_x w_y T_m(u(x)) T_n(u(y))>/<w_x w_y> for
      m, n = 1..config['mixed'][key]['n'], keyed (m, n), and the joint rate <w_x w_y>/<1>
      (w = 1 for an observable the block gives no profile, as for two Born observables).

    n_max: depth of every tower (default 12 orders for Born towers, 20 for recoil towers).

    Errors: `stat_errors` is the weighted Monte Carlo error, `scale_errors` the half-range
    over the scale rows, `errors` (and `window_errors`) their quadrature sum.
    """
    born_cfg = config.get("born", {}) or {}
    recoil_cfg = config.get("recoil", {}) or {}
    mixed_cfg = config.get("mixed", {}) or {}
    w = np.asarray(fo_events[weight_key], float)
    wsc = fo_events.get(scale_key)
    wsc = None if wsc is None else np.atleast_2d(np.asarray(wsc, float))
    Nb = n_max if n_max is not None else 12      # MOMENT_CONSTRUCTION.md, section 4
    Nr = n_max if n_max is not None else 20

    def stat_err(x, ww, nmax, a, b, mp):
        # weighted-MC error on each moment: sqrt(sum w^2 (T_n - mu_n)^2)/sum w
        u = _umap(np.clip(np.asarray(x, float), a, b), a, b, mp)
        C = _cheb(u, nmax)
        s = ww.sum()
        mu = np.array([(ww * C[:, n]).sum() / s for n in range(1, nmax + 1)])
        return np.array([np.sqrt((ww**2 * (C[:, n] - mu[n - 1])**2).sum()) / s
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

    def rate_of(wf):
        R = float((w * wf).sum() / w.sum())
        R_stat = float(np.sqrt(((w * (wf - R)) ** 2).sum()) / w.sum())
        R_scale = (0.0 if wsc is None else
                   float(0.5 * np.ptp([(wsc[r] * wf).sum() / wsc[r].sum() for r in range(wsc.shape[0])])))
        return R, R_stat, R_scale

    moments = {"born": {}, "recoil": {}}
    for o in born_cfg:
        a, b = born_cfg[o]["range"]
        mp = born_cfg[o].get("map", "lin")
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
            a = float(prof.get("a", x_match))
            b = float(prof.get("b", a))
            c = prof.get("c", xhi)
            d = prof.get("d", None)
            if x_match is not None and not np.isclose(float(x_match), a):
                warnings.warn(
                    f"compute_fo_moments: {obs} has a window starting at a={a:g}; x_match={x_match:g} is ignored")
            wf = profile_w(xr, a, b, None if c is None else float(c),
                           None if d is None else float(d))
            xm = a
        else:
            if x_match is None:
                raise ValueError(
                    f"compute_fo_moments: {obs} has no 'profile' in config, so x_match is needed")
            wf = ((xr >= x_match) & (xr < xhi)).astype(float)
            xm = float(x_match)
        sel = wf > 0
        ww = w * wf
        muw = chebyshev_moment(xr[sel], ww[sel], Nr, soft_lo, xhi, mp)
        sc = (None if wsc is None else
              0.5 * np.ptp([chebyshev_moment(xr[sel], wsc[r][sel] * wf[sel], Nr, soft_lo, xhi, mp)
                            for r in range(wsc.shape[0])], axis=0))
        dd = pack(muw, stat_err(xr[sel], ww[sel], Nr, soft_lo, xhi, mp), sc)
        R, R_stat, R_scale = rate_of(wf)
        moments["recoil"][obs] = dict(
            window_values=muw.tolist(), window_errors=dd["errors"],
            stat_errors=dd["stat_errors"], **({"scale_errors": dd["scale_errors"]} if "scale_errors" in dd else {}),
            wprofile_values=muw.tolist(), wprofile_rate=R,
            rate=R, rate_stat=R_stat, rate_scale=R_scale,
            x_match=xm, x_hi=xhi, soft_lo=soft_lo)

    if mixed_cfg:
        moments["mixed"] = {}
    for key, mc in mixed_cfg.items():
        ox, oy = mc["observables"]
        nmx = int(mc["n"])
        (ax, bx), (ay, by) = mc["range"][ox], mc["range"][oy]
        mpx = mc.get("map", {}).get(ox, "log")
        mpy = mc.get("map", {}).get(oy, "log")
        X = np.asarray(fo_events[ox], float)
        Y = np.asarray(fo_events[oy], float)
        prof = mc.get("profile") or {}
        wxy = _window_fn(X, prof.get(ox)) * _window_fn(Y, prof.get(oy))
        Cx = _cheb(_umap(np.clip(X, ax, bx), ax, bx, mpx), nmx)
        Cy = _cheb(_umap(np.clip(Y, ay, by), ay, by, mpy), nmx)
        ww = w * wxy
        S = ww.sum()
        vals, st, sc = {}, {}, {}
        for m in range(1, nmx + 1):
            for n in range(1, nmx + 1):
                f = Cx[:, m] * Cy[:, n]
                v = float((ww * f).sum() / S)
                vals[(m, n)] = v
                st[(m, n)] = float(np.sqrt((ww ** 2 * (f - v) ** 2).sum()) / S)
                if wsc is not None:
                    sc[(m, n)] = float(0.5 * np.ptp([(wsc[r] * wxy * f).sum() / (wsc[r] * wxy).sum()
                                                     for r in range(wsc.shape[0])]))
        R, R_stat, R_scale = rate_of(wxy)
        blk = dict(values=vals, stat_errors=st,
                   errors={k: float(np.hypot(st[k], sc.get(k, 0.0))) for k in vals},
                   rate=R, rate_stat=R_stat, rate_scale=R_scale)
        if wsc is not None:
            blk["scale_errors"] = sc
        moments["mixed"][key] = blk
    return moments


def combine_moments(anchor, target, n=None):
    r"""Precision-weighted combination of two moment dicts for the SAME observable and map.

    `anchor` is the precise lower-order input (e.g. the released NLO Z+jet recoil tower,
    pooled over 40 NNLOJET runs); its scale half-range per moment, `scale_errors` (and `rate_scale` for
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
                raise ValueError(
                    f"anchor and target differ in {k}: {anchor.get(k)} vs {target.get(k)}")
    vkey = "window_values" if rec else "values"
    ekey = "window_errors" if rec else "errors"
    muA = np.asarray(anchor[vkey], float)
    muT = np.asarray(target[vkey], float)
    N = min(len(muA), len(muT)) if n is None else int(n)
    muA, muT = muA[:N], muT[:N]
    sA = np.asarray(anchor.get("scale_errors", np.zeros(N)), float)[:N]
    sT = np.asarray(target.get("stat_errors", target.get(ekey)), float)[:N]
    with np.errstate(divide="ignore", invalid="ignore"):
        pA = np.where(sA > 0, 1.0 / sA ** 2, np.inf)
        pT = np.where(sT > 0, 1.0 / sT ** 2, np.inf)
        both_inf = np.isinf(pA) & np.isinf(pT)
        wT = np.where(np.isinf(pA), 0.0, np.where(np.isinf(pT), 1.0, pT / (pA + pT)))
        wT = np.where(both_inf, 0.5, wT)
        s2 = np.where(np.isinf(pA) | np.isinf(pT), 0.0, 1.0 / (pA + pT))
    mu = (1.0 - wT) * muA + wT * muT
    s = np.sqrt(s2)
    out = dict(target)
    out[vkey] = mu.tolist()
    out["stat_errors"] = s.tolist()
    out["scale_errors"] = sA.tolist()
    out[ekey] = np.hypot(s, sA).tolist()
    meta = dict(n=int(N), target_weight=wT.tolist())
    if rec:
        out["wprofile_values"] = out[vkey]
        RA = float(anchor.get("wprofile_rate", anchor["rate"]))
        RT = float(target.get("wprofile_rate", target["rate"]))
        rA = float(anchor.get("rate_scale", 0.0) or 0.0)
        rT = float(target.get("rate_stat", 0.0) or 0.0)
        if rA > 0 and rT > 0:
            pa, pt = 1.0 / rA ** 2, 1.0 / rT ** 2
            wR = pt / (pa + pt)
            R = (1 - wR) * RA + wR * RT
            rs = (pa + pt) ** -0.5
        elif rA == 0:
            wR, R, rs = 0.0, RA, 0.0
        else:
            wR, R, rs = 1.0, RT, 0.0
        out["rate"] = out["wprofile_rate"] = float(R)
        out["rate_stat"] = float(rs)
        out["rate_scale"] = rA
        meta.update(rate_target_weight=float(wR), rate_anchor=RA, rate_target=RT)
    out["_combine"] = meta
    return out
