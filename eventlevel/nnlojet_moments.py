#!/usr/bin/env python3
r"""Event-level fixed-order moments from NNLOJET PROFILE histograms.

NNLOJET is patched so that a single-bin histogram booked with `profile = T` of a
Chebyshev-moment observable chebT_<x>_n accumulates the literal per-phase-space
point sum

    prof_<x>_n = sum_j w_j T_n(u(x_j)) / binwidth ,

a genuine event-level weighted moment, NOT a moment reconstructed from a binned
distribution.  A companion NORMAL single-bin histogram of the same observable
accumulates

    norm = sum_j w_j / binwidth = sigma / binwidth ,

so the ratio prof/norm = <T_n> exactly (the bin width cancels).  Channels are
VEGAS estimates of int T_n dsigma_channel and hence ADD; several independent
seeds give the statistical error and the 7-point scale set gives the scale band.

This module reads those .dat files and returns the `moments` dict consumed by
`maxent_upgrade.upgrade` -- i.e. it is the event-level replacement for the
histogram-derived `compute_fo_moments`.
"""
import glob
import os

import numpy as np


def _load(path):
    """NNLOJET .dat -> (lower, center, upper, vals[nbin,nscale], errs[nbin,nscale])."""
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append([float(x) for x in line.split()])
    a = np.array(rows, float)
    rest = a[:, 3:]
    return a[:, 0], a[:, 1], a[:, 2], rest[:, 0::2], rest[:, 1::2]


def _sum_channels(base, run, channels, tag, seed, prefix="Z"):
    """Additive channel combination of a single-bin profile/normal value array.

    Returns vals[nscale] summed over channels (single bin), or None if missing.
    """
    tot = None
    for ch in channels:
        r = _load(os.path.join(base, f"ch_{ch}", f"{prefix}.{run}.{ch}.{tag}.s{seed}.dat"))
        if r is None:
            continue
        v = r[3][0]                      # single bin -> [nscale]
        tot = v.copy() if tot is None else tot + v
    return tot


def fo_curve(base, run, channels, seeds, tag, prefix="Z", scale_idx=0, rebin=1):
    r"""Channel-summed, seed-pooled fixed-order DISTRIBUTION with an honest error.

    Returns ``(lo, hi, dens, err)`` where ``dens`` is the differential cross
    section per bin and ``err`` its uncertainty, or ``None`` if nothing loads.

    The uncertainty is the **seed-to-seed scatter** of the channel sum,
    ``std(t_s, ddof=1)/sqrt(nseed)``, NOT the error column in the .dat files.
    For multi-bin histograms of a subtracted calculation those quoted per-bin
    errors are wildly wrong: for the DY ``abs_yl1`` histogram at 40 seeds the
    quoted error exceeds the actual seed scatter by up to a factor of 1.3e6
    (V channel: quoted 1.85e6 against a scatter of 10.6), because the huge
    point-by-point fluctuations of the R-V subtraction cancel between bins in a
    way the per-bin error does not track.  The integrals are meanwhile exact --
    ``yl1_a`` reproduces ``norm_born`` channel by channel.  Masking bins on the
    quoted error therefore throws away perfectly good data; the across-seed
    scatter is the honest estimator, and is what running many seeds buys.

    ``rebin`` merges groups of adjacent bins first (the moments are integrals
    and do not care about binning; this is only a reference curve).
    """
    per_seed = []
    lo = hi = None
    for s in seeds:
        tot = None
        for ch in channels:
            r = _load(os.path.join(base, f"ch_{ch}", f"{prefix}.{run}.{ch}.{tag}.s{s}.dat"))
            if r is None:
                continue
            lo, _, hi, v, _ = r
            tot = v[:, scale_idx].copy() if tot is None else tot + v[:, scale_idx]
        if tot is not None:
            per_seed.append(tot)
    if not per_seed or lo is None:
        return None
    A = np.asarray(per_seed, float)              # (nseed, nbin) densities
    w = hi - lo
    K = max(int(rebin), 1)
    if K > 1:
        nb = (A.shape[1] // K) * K
        lo, hi = lo[:nb].reshape(-1, K)[:, 0], hi[:nb].reshape(-1, K)[:, -1]
        A = (A[:, :nb] * w[:nb]).reshape(A.shape[0], -1, K).sum(2)   # integrals
        w = hi - lo
        A = A / w                                                     # back to density
    n = A.shape[0]
    dens = A.mean(0)
    err = A.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.full_like(dens, np.nan)
    return lo, hi, dens, err


# ---------------------------------------------------------------------------
# Which booked histograms live on a MIRRORED axis relative to the observable
# the analysis works with.  This has now caused the same bug three times --
# twice in the figures and once in the Z+jet scan -- because the convention
# lived in each consumer's head instead of in one place.  Anything reading a
# fixed-order curve should go through `oriented_fo_curve`, which consults this
# registry, rather than remembering to mirror by hand.
#
#   tag         booked as              analysis observable
#   dphi_aa     pi_dphi_g1g2 (pi-dphi) dphi_aa   (Delta phi)
#   dphil_a     dphi_l1l2    (dphi)    pimdphi   (pi - Delta phi)
#
# Both map x -> pi - x; the density is unchanged under it (unit Jacobian), so
# only the edges reverse.
# ---------------------------------------------------------------------------
MIRRORED_TAGS = {"dphi_aa", "dphil_a"}


def oriented_fo_curve(base, run, channels, seeds, tag, prefix="Z", **kw):
    r"""`fo_curve` with the axis put on the ANALYSIS observable's orientation.

    Returns (lo, hi, dens, err) exactly like `fo_curve`, but for any tag in
    MIRRORED_TAGS the axis has been mapped x -> pi - x and re-sorted, so the
    caller never has to know how NNLOJET happened to book it.
    """
    r = fo_curve(base, run, channels, seeds, tag, prefix=prefix, **kw)
    if r is None:
        return None
    lo, hi, dens, err = r
    if tag in MIRRORED_TAGS:
        lo, hi = (np.pi - hi)[::-1], (np.pi - lo)[::-1]
        dens, err = dens[::-1], err[::-1]
    return lo, hi, dens, err


def _moment_over_seeds(base, run, channels, seeds, prof_tag, norm_tag, prefix="Z"):
    r"""Combine seeds CORRECTLY, by pooling the Monte Carlo.

    Every seed is an independent estimate of the same two integrals, so the
    unbiased combination is the ratio of the SUMS,

        <T_n> = (sum_seeds sum_ch prof) / (sum_seeds sum_ch norm) ,

    not the mean of the per-seed ratios (a ratio of means is not the mean of
    ratios; one seed with a small denominator otherwise throws a wild value into
    the average and can make the moment set infeasible).

    Returns (pooled[nscale], per_seed[nseed, nscale]); the per-seed ratios are
    kept only to estimate the statistical spread.
    """
    P = N = None
    rows = []
    for s in seeds:
        prof = _sum_channels(base, run, channels, prof_tag, s, prefix)
        norm = _sum_channels(base, run, channels, norm_tag, s, prefix)
        if prof is None or norm is None:
            continue
        P = prof.copy() if P is None else P + prof
        N = norm.copy() if N is None else N + norm
        with np.errstate(divide="ignore", invalid="ignore"):
            rows.append(prof / norm)
    if P is None:
        return None, np.zeros((0, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        pooled = P / N
    return pooled, np.array(rows)


def _reduce(m, scale_idx=0):
    """(pooled[nscale], per_seed[nseed,nscale]) -> (central, stat, scale, total)."""
    pooled, per_seed = m
    if pooled is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    central = float(pooled[scale_idx])                    # pooled MC, unbiased
    ns = len(per_seed)
    if ns > 1:
        v = per_seed[:, scale_idx]
        v = v[np.isfinite(v)]
        stat = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    else:
        stat = 0.0
    scale = float(0.5 * (np.nanmax(pooled) - np.nanmin(pooled)))
    return central, stat, scale, float(np.hypot(stat, scale))


def seeds_in(base, run, ref_channel="LO", ref_tag="norm_born"):
    ss = set()
    for f in glob.glob(os.path.join(base, f"ch_{ref_channel}",
                                    f"Z.{run}.{ref_channel}.{ref_tag}.s*.dat")):
        ss.add(int(f.split(".s")[-1].split(".dat")[0]))
    return sorted(ss)



def common_seeds(base, run, channels, tag="norm_born", prefix="Z"):
    r"""Seeds for which EVERY channel is present, for EVERY requested tag.

    Pooling requires the same channel content in every seed: if one channel is
    missing for some seed, that seed contributes an incomplete sum and biases the
    pooled moment (e.g. a missing Born channel wrecks the Born denominator).

    `tag` may be a single histogram name or a sequence of them.  Always pass the
    tags you are actually going to sum: selecting seeds on one tag and summing a
    different one silently reintroduces the incomplete-sum bias, because a seed
    can have written the first histogram and not the second.
    """
    tags = [tag] if isinstance(tag, str) else list(tag)
    sets = []
    for tg in tags:
        for ch in channels:
            ss = set()
            for f in glob.glob(os.path.join(base, f"ch_{ch}",
                                            f"{prefix}.{run}.{ch}.{tg}.s*.dat")):
                ss.add(int(f.split(".s")[-1].split(".dat")[0]))
            sets.append(ss)
    return sorted(set.intersection(*sets)) if sets else []


def fo_moments_from_nnlojet(base, run, channels, seeds, born_tags, recoil_tag,
                            n_born, n_recoil, x_match, x_hi, soft_lo,
                            recoil_cfg_name=None, scale_idx=0,
                            norm_born="norm_born", norm_recoil="norm_ptzwin"):
    """scale_idx selects the scale column for the CENTRAL moment values
    (0 = central scale); pass 1..6 to build the per-scale moment sets whose
    re-solves form the output scale band."""
    r"""Build the `moments` dict for `maxent_upgrade.upgrade` from NNLOJET profiles.

    Parameters
    ----------
    base : directory holding the per-channel subdirs ch_<CH>/
    run  : NNLOJET RUN name (the <RUN> in Z.<RUN>.<CH>.<hist>.s<seed>.dat)
    channels : list of channel labels to sum, e.g. ['LO','R','V'] (NLO) or the six
               NNLO labels.  Their profiles add (linear in the cross section).
    seeds : list of iseed values to average (statistical error from their spread)
    born_tags : {config_obs_name: nnlojet_obs_tag}, e.g. {'mll':'mll','y_abs':'absyz'}
    recoil_tag : nnlojet observable tag for the recoil, e.g. 'ptz'
    n_born, n_recoil : number of Chebyshev orders booked for Born / recoil
    x_match, x_hi, soft_lo : the recoil window edges and the log-map floor, echoed
        into the recoil dict (must match the booked selector + eval map).

    Returns
    -------
    moments : the dict consumed by `upgrade`:
        {'born':   {obs: {'values':[..], 'errors':[..]}},
         'recoil': {obs: {'window_values':[..], 'window_errors':[..],
                          'rate':R, 'x_match':xm, 'x_hi':xh, 'soft_lo':s}}}
    """
    moments = {"born": {}, "recoil": {}}

    for cfg_obs, tag in born_tags.items():
        vals, errs = [], []
        for n in range(1, n_born + 1):
            m = _moment_over_seeds(base, run, channels, seeds,
                                   f"prof_{tag}_{n}", norm_born)
            c, st, sc, tot = _reduce(m, scale_idx)
            vals.append(c); errs.append(tot)
        moments["born"][cfg_obs] = dict(values=vals, errors=errs)

    # recoil: moments over the window, plus the FO window rate sigma_win/sigma_fid
    vals, errs = [], []
    for n in range(1, n_recoil + 1):
        m = _moment_over_seeds(base, run, channels, seeds,
                               f"prof_{recoil_tag}_{n}", norm_recoil)
        c, st, sc, tot = _reduce(m, scale_idx)
        vals.append(c); errs.append(tot)
    # rate = sigma_window / sigma_fiducial (bin widths identical -> ratio of norms)
    rate_per_seed = []
    for s in seeds:
        nw = _sum_channels(base, run, channels, norm_recoil, s)
        nb = _sum_channels(base, run, channels, norm_born, s)
        if nw is not None and nb is not None:
            rate_per_seed.append(nw[scale_idx] / nb[scale_idx])
    rate = float(np.mean(rate_per_seed)) if rate_per_seed else float("nan")

    # key by the config observable name (e.g. 'pT_ll'), not the NNLOJET tag ('ptz')
    key = recoil_cfg_name if recoil_cfg_name is not None else recoil_tag
    moments["recoil"][key] = dict(
        window_values=vals, window_errors=errs, rate=rate,
        x_match=float(x_match), x_hi=float(x_hi), soft_lo=float(soft_lo))
    return moments


def fo_moments_smooth_from_nnlojet(base, run, channels, seeds, born_tags,
                                   n_born, n_recoil, x_match, x_hi, soft_lo,
                                   recoil_cfg_name="pT_ll", scale_idx=0,
                                   norm_born="norm_born", w0="prof_wptz_0",
                                   wtag="prof_wptz", prefix="Z"):
    r"""Like fo_moments_from_nnlojet but with the SMOOTH profile recoil target,
    computed event-level in NNLOJET:  <T_n>_w = prof_wptz_n / prof_wptz_0  and
    R = prof_wptz_0 / sigma_fiducial.  The profile w(pT) is baked into the eval
    (must match maxent_upgrade.profile_w in the solver config).
    """
    moments = {"born": {}, "recoil": {}}
    for cfg_obs, tag in born_tags.items():
        vals, errs = [], []
        for n in range(1, n_born + 1):
            m = _moment_over_seeds(base, run, channels, seeds, f"prof_{tag}_{n}", norm_born, prefix)
            c, st, sc, tot = _reduce(m, scale_idx); vals.append(c); errs.append(tot)
        moments["born"][cfg_obs] = dict(values=vals, errors=errs)

    vals, errs = [], []
    for n in range(1, n_recoil + 1):
        m = _moment_over_seeds(base, run, channels, seeds, f"{wtag}_{n}", w0, prefix)
        c, st, sc, tot = _reduce(m, scale_idx); vals.append(c); errs.append(tot)
    rate_ps = []
    for s in seeds:
        wnum = _sum_channels(base, run, channels, w0, s, prefix)
        nb = _sum_channels(base, run, channels, norm_born, s, prefix)
        if wnum is not None and nb is not None:
            rate_ps.append(wnum[scale_idx] / nb[scale_idx])
    R = float(np.mean(rate_ps)) if rate_ps else float("nan")

    moments["recoil"][recoil_cfg_name] = dict(
        window_values=vals, window_errors=errs,
        wprofile_values=vals, wprofile_rate=R, rate=R,
        x_match=float(x_match), x_hi=float(x_hi), soft_lo=float(soft_lo))
    return moments


def add_profiled_recoil(moments, base, run, channels, seeds, cfg_obs,
                        wtag, w0, n_recoil, x_match, x_hi, soft_lo,
                        norm_born="norm_born", scale_idx=0, prefix="Z"):
    r"""Attach a SECOND (or third) profiled recoil observable to `moments`.

    Same construction as the primary recoil -- <T_n>_w = prof_<wtag>_n / prof_w0
    and R = prof_w0 / sigma_fiducial -- for an observable whose own profile is
    compiled into NNLOJET.  Used for the diphoton pi-dphi constraint: pT alone
    leaves the shower's pT<->dphi correlation wrong, and pi-dphi is a recoil
    observable, so it must enter through a profile rather than over its full
    range (which makes the dual infeasible).

    The profile parameters passed to the solver MUST mirror the Fortran
    (eval_w_dpa: pa=0.3, pb=0.6, no upper cut).
    """
    vals, errs = [], []
    for n in range(1, n_recoil + 1):
        m = _moment_over_seeds(base, run, channels, seeds, f"{wtag}_{n}", w0, prefix)
        c, st, sc, tot = _reduce(m, scale_idx); vals.append(c); errs.append(tot)
    rate_ps = []
    for s in seeds:
        wnum = _sum_channels(base, run, channels, w0, s, prefix)
        nb = _sum_channels(base, run, channels, norm_born, s, prefix)
        if wnum is not None and nb is not None:
            rate_ps.append(wnum[scale_idx] / nb[scale_idx])
    R = float(np.mean(rate_ps)) if rate_ps else float("nan")
    moments["recoil"][cfg_obs] = dict(
        window_values=vals, window_errors=errs,
        wprofile_values=vals, wprofile_rate=R, rate=R,
        x_match=float(x_match), x_hi=float(x_hi), soft_lo=float(soft_lo))
    return moments


def add_mixed_moments(moments, base, run, channels, seeds, key, wtag, w0,
                      n_max, norm_born="norm_born", scale_idx=0, prefix="Z"):
    r"""Attach MIXED two-observable moments  <T_m(x) T_n(y)>_w  to `moments`.

    NNLOJET books these as prof_<wtag>_<m><n> (orders packed as a two-digit
    index, matching eval_chebTT_*), with prof_<w0> the joint w-rate numerator.
    Separate moments of x and y constrain only the two marginals; the mixed
    ones constrain the joint distribution, which is what a follower determined
    by the CORRELATION depends on.
    """
    vals = {}
    for m in range(1, n_max + 1):
        for n in range(1, n_max + 1):
            mm = _moment_over_seeds(base, run, channels, seeds,
                                    f"{wtag}_{m}{n}", w0, prefix)
            if mm is None:
                continue
            vals[(m, n)] = _reduce(mm, scale_idx)[0]
    rate_ps = []
    for s in seeds:
        wnum = _sum_channels(base, run, channels, w0, s, prefix)
        nb = _sum_channels(base, run, channels, norm_born, s, prefix)
        if wnum is not None and nb is not None:
            rate_ps.append(wnum[scale_idx] / nb[scale_idx])
    R = float(np.mean(rate_ps)) if rate_ps else float("nan")
    moments.setdefault("mixed", {})[key] = dict(values=vals, rate=R)
    return moments


if __name__ == "__main__":
    import json
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/user/nnlojet-v1.0.2/dy_profile_poc"
    run = "DY_MOMENTS"
    channels = ["LO", "R", "V"]
    seeds = seeds_in(base, run)
    print(f"seeds: {seeds}", file=sys.stderr)
    mom = fo_moments_from_nnlojet(
        base, run, channels, seeds,
        born_tags={"mll": "mll", "y_abs": "absyz"}, recoil_tag="ptz",
        recoil_cfg_name="pT_ll",
        n_born=6, n_recoil=12, x_match=10.0, x_hi=500.0, soft_lo=0.5)
    print(json.dumps(mom, indent=2))
