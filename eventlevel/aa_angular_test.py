#!/usr/bin/env python3
r"""Does adding |cos theta*| to the constraint set recover it -- and at what cost?

Runs the identical diphoton upgrade twice on the SAME events and the SAME FO
moments, differing only in the Born constraint set:

  A  mass-only : Born {m_aa}                  + recoil {pt_aa}
  B  mass+cos  : Born {m_aa, |cos theta*|}    + recoil {pt_aa}

and prints the median |shape/data - 1| against ATLAS 1704.03839 for every
measured observable, so the question "does constraining the angle recover
cos theta* AND leave m_aa alone?" is answered by one table.

pi - dphi is NOT offered as a third variant, and that is a result rather than an
omission: it is a RECOIL observable (identically zero at Born level, Sudakov-
divergent as dphi -> pi), so imposing its fixed-order moments over the full
range contradicts the recoil profile -- which deliberately keeps the shower
below the seam -- and the dual has no positive-weight solution at ANY moment
order (verified for N = 3, 4, 6).  |cos theta*| is a genuine Born observable and
converges at all of them.  Which slot an observable belongs in is dictated by
whether it is defined and infrared-safe at Born level, not by preference.

The covariance identity  <y>_q - <y>_p = Cov_p(r, y)/<r>_p  is also evaluated
per observable, which tells you *why* a follower moved.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from maxent_upgrade import upgrade, check_seam
from nnlojet_moments import (fo_moments_smooth_from_nnlojet, common_seeds,
                             add_profiled_recoil)
from aa_vs_data import load_prior_full

GGDIR = os.environ.get("GGDIR", "/Users/user/nnlojet-v1.0.2/gg_moments")
RUN, PREFIX = "GG_MOMENTS", "GG"
CH = ["LO", "R", "V", "RR", "RV", "VV"]
# mirrors eval_w_ptaa (pa=28, pb=56, pc=pd=500)
XM, XB, XHI, SOFT = 28.0, 56.0, 500.0, 1.0
Q_HARD = 90.0
MLO, MHI = 80.0, 700.0
NEV = int(os.environ.get("NEV", 1_200_000))

D = dict(np.load(os.path.join(HERE, "atlas_aa_8tev.npz"), allow_pickle=True))
ROWS = [("m_aa", "m_aa"), ("pt_aa", "pt_aa"), ("costh_aa", "costh_aa"),
        ("dphi_aa", "dphi_aa"), ("at_aa", "at_aa"), ("y_aa", "y_aa")]

# constraint sets compared side by side (all share the same recoil profile)
# Born constraint set, and whether pi-dphi is ALSO constrained as a profiled
# RECOIL observable.  Constraining pT alone leaves the shower's pT<->dphi
# correlation wrong: MaxEnt then overshoots data AND fixed order by ~60% near
# dphi = 2.2 rad, where the prior has ample statistics (N_eff ~ 1e4), so it is
# not a support artefact.  pi-dphi cannot be a BORN constraint (infeasible at
# any moment order), so it enters through its own compiled profile.
VARIANTS = [("A mass-only", ("m_aa",), False),
            ("B mass+cos", ("m_aa", "costh_aa"), False),
            ("C +dphi recoil", ("m_aa", "costh_aa"), True)]
CONSTRAINED = {t: set(o) | {"pt_aa"} | ({"dphi_aa"} if d else set())
               for t, o, d in VARIANTS}
# mirrors eval_w_dpa in EvalFuncs.f90
DP_A, DP_B, DP_LO, DP_HI = 0.3, 0.6, 0.01, np.pi
N_DPA = int(os.environ.get('N_DPA', 3))   # pi-dphi moments to impose


def dens(x, w, e):
    h, _ = np.histogram(x, e, weights=w / w.sum())
    return h / np.diff(e)


def dev(x, w, key):
    """median |shape/data - 1| in %, or nan if the observable is not measured."""
    if f"{key}_val" not in D:
        return np.nan, 0
    lo = np.asarray(D[f"{key}_lo"], float); hi = np.asarray(D[f"{key}_hi"], float)
    val = np.asarray(D[f"{key}_val"], float)
    e = np.concatenate([lo[:1], hi])
    if not np.all(np.diff(e) > 0):
        return np.nan, 0
    h = dens(x, w, e)
    m = (val > 0) & (h > 0)
    if m.sum() == 0:
        return np.nan, 0
    bw = np.diff(e)[m]
    hh = h[m] / (h[m] * bw).sum(); dd = val[m] / (val[m] * bw).sum()
    return 100 * np.median(np.abs(hh / dd - 1)), int(m.sum())


def build_events():
    ev = load_prior_full()
    n = len(ev["weight"])
    idx = np.random.default_rng(0).choice(n, min(NEV, n), replace=False)
    ev = {k: v[idx] for k, v in ev.items()}
    # the solver needs the same names the NNLOJET moments were booked under
    ev["y_abs"] = ev["y_aa"]
    ev["dphi_log"] = np.maximum(np.pi - ev["dphi_aa"], 1e-8)
    return ev, idx


BORN_CFG = {"m_aa": {"range": (MLO, MHI), "map": "log"},
            "costh_aa": {"range": (0.0, 1.0), "map": "lin"},
            # NOTE pi-dphi is a RECOIL observable, not a Born one: it vanishes
            # identically at Born level and the fixed-order prediction is
            # Sudakov-divergent as dphi -> pi.  Imposing it over its full range
            # contradicts the recoil profile (which deliberately keeps the
            # shower below the seam) and the dual has no positive-weight
            # solution.  It is therefore left as a prediction.
            "dphi_log": {"range": (0.01, np.pi), "map": "log"}}


def solve(ev, M, obs, dphi_recoil=False):
    born = {k: BORN_CFG[k] for k in obs}
    recoil = {"pt_aa": {"range": (SOFT, XHI), "map": "log", "soft_lo": SOFT,
                        "profile": {"a": XM, "b": XB, "c": XHI}}}
    if dphi_recoil:
        recoil["dphi_log"] = {"range": (DP_LO, DP_HI), "map": "log",
                              "soft_lo": DP_LO,
                              "profile": {"a": DP_A, "b": DP_B}}
    cfg = dict(born=born, recoil=recoil, followers=["y_abs"],
               moment_selection=False)
    return upgrade(ev, M, cfg)


def main():
    check_seam(XM, Q_HARD, label="diphoton")
    born_tags = {"m_aa": "maa", "costh_aa": "cts", "dphi_log": "dpa"}
    # select seeds on EVERY tag that is summed, not just the Born denominator
    need = ["norm_born", "prof_wpt_0"] + [f"prof_{t}_1" for t in born_tags.values()] \
           + [f"prof_wpt_{n}" for n in range(1, 13)]
    seeds = common_seeds(GGDIR, RUN, CH, tag=need, prefix=PREFIX)
    print(f"FO seeds (channel- and tag-complete): {len(seeds)}"
          f"  -> {seeds[:8]}{' ...' if len(seeds) > 8 else ''}")
    if not seeds:
        sys.exit("no complete diphoton seeds")

    M = fo_moments_smooth_from_nnlojet(
        GGDIR, RUN, CH, seeds, born_tags=born_tags, n_born=6, n_recoil=12,
        x_match=XM, x_hi=XHI, soft_lo=SOFT, recoil_cfg_name="pt_aa",
        norm_born="norm_born", w0="prof_wpt_0", wtag="prof_wpt", prefix=PREFIX)
    # second profiled recoil: pi - dphi (only if those moments were produced)
    have_dpa = bool(common_seeds(GGDIR, RUN, CH, tag="prof_wdpa_0", prefix=PREFIX))
    if have_dpa:
        sd = common_seeds(GGDIR, RUN, CH,
                          tag=["prof_wdpa_0"] + [f"prof_wdpa_{n}" for n in range(1, N_DPA + 1)],
                          prefix=PREFIX)
        add_profiled_recoil(M, GGDIR, RUN, CH, sd, "dphi_log",
                            wtag="prof_wdpa", w0="prof_wdpa_0", n_recoil=N_DPA,
                            x_match=DP_A, x_hi=DP_HI, soft_lo=DP_LO, prefix=PREFIX)
        print(f"  pi-dphi recoil moments: {len(sd)} seeds  "
              f"<T_n>_w = {' '.join(f'{v:+.4f}' for v in M['recoil']['dphi_log']['window_values'])}"
              f"  R = {M['recoil']['dphi_log']['rate']:.4f}")
    else:
        print("  pi-dphi recoil moments NOT present -- variant C will be skipped")
    for o, t in born_tags.items():
        v = M["born"][o]["values"]
        e = M["born"][o].get("errors")
        s = " ".join(f"{x:+.4f}" for x in v)
        print(f"  <T_n({o:9s})> = {s}")
        if e is not None and len(e):
            print(f"          err   = {' '.join(f'{x:8.4f}' for x in e)}")

    ev, idx = build_events()
    print(f"prior events: {len(ev['weight']):,}\n")

    out = {}
    for tag, obs, dpr in VARIANTS:
        if dpr and not have_dpa:
            out[tag] = None; print(f"{tag:16s}  skipped (no pi-dphi moments yet)"); continue
        try:
            r = solve(ev, M, obs, dphi_recoil=dpr)
            out[tag] = r
            print(f"{tag:16s}  effN {100*r.effN:5.1f}%   closure {r.closure:.2e}   "
                  f"neg-wt {100*np.mean(r.weights <= 0):.1f}%")
        except Exception as exc:                                  # noqa: BLE001
            out[tag] = None
            print(f"{tag:16s}  FAILED: {exc}")

    # ---- table against ATLAS -------------------------------------------------
    wpr = ev["weight"]
    cols = "".join(f" | {t:>14}" for t, _, _ in VARIANTS)
    hdr = f"\n{'observable':>10} | {'prior':>7}{cols} | bins"
    print(hdr); print("-" * len(hdr))
    for key, dkey in ROWS:
        f_pr, nb = dev(ev[key], wpr, dkey)
        if not np.isfinite(f_pr):
            continue
        cells = []
        for tag, _, _ in VARIANTS:
            r = out[tag]
            if r is None:
                cells.append(f"{'--':>14}")
            else:
                f, _ = dev(ev[key], r.weights, dkey)
                mark = "c" if key in CONSTRAINED[tag] else " "
                cells.append(f"{f:12.1f}%{mark}")
        print(f"{key:>10} | {f_pr:6.1f}% | " + " | ".join(cells) + f" | {nb:4d}")
    print("\n  'c' marks an observable that was CONSTRAINED in that variant.")

    # ---- covariance identity: why did each follower move? --------------------
    # Ship variant B.  Variant C (pi-dphi constrained as a profiled recoil) is
    # kept in the table as a DOCUMENTED NEGATIVE RESULT: it improves the local
    # excursion near dphi = 2.2 rad (peak |ratio-1| 83% -> 23-34% depending on
    # moment count) but degrades agreement with ATLAS overall, at every moment
    # count tried (N = 2, 3, 6), and costs an order of magnitude in effective
    # statistics.  The reason is precision, not principle: the fixed-order
    # pi-dphi moments carry errors 0.024-0.069 against 0.002-0.007 for
    # |cos theta*|, so imposing them injects more noise than information.
    # Constraining dphi properly needs more fixed-order statistics, not a
    # different constraint.
    saved = "B mass+cos" if out.get("B mass+cos") is not None else \
        next((t for t, *_ in reversed(VARIANTS) if out.get(t) is not None), None)
    r = out.get(saved) if saved else None
    if r is not None:
        p = wpr / wpr.sum(); q = r.weights / r.weights.sum()
        ratio = q / np.maximum(p, 1e-300)
        rbar = float((p * ratio).sum())
        print(f"\ncovariance identity   <y>_q - <y>_p = Cov_p(r,y)/<r>_p   (<r>_p = {rbar:.6f})")
        print(f"{'observable':>10} | {'<y>_p':>10} | {'<y>_q':>10} | {'Cov/<r>':>10} | {'exact?':>8}")
        for key, _ in ROWS:
            y = ev[key]
            yp = float((p * y).sum()); yq = float((q * y).sum())
            cov = float((p * (ratio - rbar) * (y - yp)).sum()) / rbar
            err = abs((yq - yp) - cov) / max(abs(yq - yp), 1e-12)
            print(f"{key:>10} | {yp:10.4f} | {yq:10.4f} | {cov:+10.4f} | {err:8.1e}")

        np.savez(os.path.join(HERE, "aa_eventlevel_weights.npz"),
                 weights=r.weights, idx=idx)
        print(f"\nsaved aa_eventlevel_weights.npz (variant {saved})")


if __name__ == "__main__":
    main()
