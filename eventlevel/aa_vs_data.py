#!/usr/bin/env python3
r"""Diphoton (event-level moments) vs ATLAS 1704.03839.

Constrained: m_aa, pt_aa.  Everything else is a PREDICTION -- and |cos theta*|
and y_aa are essentially uncorrelated with the constrained recoil, so they are
the referee-proof cases (measured, and not fit).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = dict(np.load(os.path.join(HERE, "atlas_aa_8tev.npz"), allow_pickle=True))


def load_prior_full():
    parts = []
    for f in ("aa_prior_s11.npz", "aa_prior_s12.npz"):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            parts.append(dict(np.load(p, allow_pickle=True)))
    g = lambda k: np.concatenate([np.asarray(p[k], float) for p in parts])
    g1, g2 = g("g1"), g("g2")
    ev = dict(m_aa=g("m_aa"), pt_aa=g("pt_aa"), y_aa=np.abs(g("y_aa")),
              dphi_aa=np.abs(g("dphi_aa")), weight=g("weight"))
    # Collins-Soper |cos theta*| and a_T from the photon four-vectors
    p1p = (g1[:, 3] + g1[:, 2]) / np.sqrt(2); p1m = (g1[:, 3] - g1[:, 2]) / np.sqrt(2)
    p2p = (g2[:, 3] + g2[:, 2]) / np.sqrt(2); p2m = (g2[:, 3] - g2[:, 2]) / np.sqrt(2)
    pz = g1[:, 2] + g2[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        cs = np.abs(np.sign(pz) * 2 * (p1p * p2m - p1m * p2p)
                    / (ev["m_aa"] * np.sqrt(ev["m_aa"] ** 2 + ev["pt_aa"] ** 2)))
    ev["costh_aa"] = np.clip(np.nan_to_num(cs), 0, 1)
    # a_T : component of pT_aa transverse to the photon thrust axis
    dx = g1[:, 0] - g2[:, 0]; dy = g1[:, 1] - g2[:, 1]
    tn = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx, ty = np.where(tn > 0, dx / tn, 0.0), np.where(tn > 0, dy / tn, 0.0)
    px, py = g1[:, 0] + g2[:, 0], g1[:, 1] + g2[:, 1]
    ev["at_aa"] = np.abs(px * ty - py * tx)
    m = np.isfinite(ev["m_aa"]) & (ev["m_aa"] > 0) & np.isfinite(ev["weight"]) & (ev["weight"] > 0)
    return {k: v[m] for k, v in ev.items()}


def dens(x, w, e):
    h, _ = np.histogram(x, e, weights=w / w.sum())
    return h / np.diff(e)


def main():
    W = dict(np.load(os.path.join(HERE, "aa_eventlevel_weights.npz")))
    wnew, idx = W["weights"], W["idx"]
    ev = load_prior_full()
    ev = {k: v[idx] for k, v in ev.items()}
    wpr = ev["weight"]

    rows = [("m_aa", "m_aa", "CONSTRAINED"), ("pt_aa", "pt_aa", "CONSTRAINED"),
            ("costh_aa", "costh_aa", "predicted (rho~0)"),
            ("dphi_aa", "dphi_aa", "predicted"),
            ("at_aa", "at_aa", "predicted")]
    print(f"{'observable':>10} | {'role':>18} | {'prior':>7} | {'event-level':>11} | bins")
    print("-" * 66)
    for key, dkey, role in rows:
        if f"{dkey}_val" not in D or key not in ev:
            continue
        lo, hi = np.asarray(D[f"{dkey}_lo"], float), np.asarray(D[f"{dkey}_hi"], float)
        val = np.asarray(D[f"{dkey}_val"], float)
        e = np.concatenate([lo[:1], hi])
        if not np.all(np.diff(e) > 0):
            continue
        hp, hq = dens(ev[key], wpr, e), dens(ev[key], wnew, e)
        # shape comparison: normalise data and MC over the same range
        dv = val / (val * np.diff(e)).sum()
        m = (dv > 0) & (hp > 0)
        if m.sum() == 0:
            continue
        pp = hp[m] / (hp[m] * np.diff(e)[m]).sum()
        qq = hq[m] / (hq[m] * np.diff(e)[m]).sum()
        dd = dv[m] / (dv[m] * np.diff(e)[m]).sum()
        f_pr = 100 * np.median(np.abs(pp / dd - 1))
        f_nw = 100 * np.median(np.abs(qq / dd - 1))
        flag = "  <-- improved" if f_nw < f_pr else "  <-- WORSE"
        print(f"{key:>10} | {role:>18} | {f_pr:6.1f}% | {f_nw:10.1f}% | {m.sum():4d}{flag}")


if __name__ == "__main__":
    main()
