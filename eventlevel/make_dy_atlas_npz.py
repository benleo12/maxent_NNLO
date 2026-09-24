#!/usr/bin/env python3
r"""Build the ATLAS-fiducial generator npz (with scale weights) from a
showered parent npz.  CHECKED-IN replacement for the throwaway inline recuts
that produced dy_mcatnlo_atlas.npz / dy_powheg_atlas.npz without w_scale.

The recut is the frozen recipe (identical to fig_suite.recut):
    66 <= mll <= 116,  pT(l+-) > 27,  |eta(l+-)| < 2.5
    phistar = tan((pi-|dphi|)/2) / cosh((eta- - eta+)/2)

Scale weights are stored as RATIOS times the shower weight,
    w_scale[:, k] = w * rwgt_k / rwgt_0,
which cancels the per-event replay noise of POWHEG's compute_rwgt and the
per-mille XWGTUP/rwgt numerics of MG5.  Columns are the 7-point set; the
(muR, muF) of every column is stored in w_scale_labels.  For an MG5 parent
the id -> (muR, muF) map is parsed from the LHE header, never assumed.

Usage:
  make_dy_atlas_npz.py --parent dy_psNLO_partons_rwl.npz \
      --lhe dy_NLO_500k.lhe.gz --out dy_mcatnlo_atlas.npz \
      [--old dy_mcatnlo_atlas.npz.bak]      # validation reference
  make_dy_atlas_npz.py --parent dy_psNLO_powheg_ct18_rwl.npz \
      --powheg --out dy_powheg_atlas.npz [--old ...]
"""
import argparse
import gzip
import os
import re

import numpy as np

SEVEN = [(1.0, 1.0), (2.0, 1.0), (0.5, 1.0), (1.0, 2.0),
         (2.0, 2.0), (1.0, 0.5), (0.5, 0.5)]
# converter lhe_legacy_to_lhef3.py id map (POWHEG chain)
POWHEG_IDS = {(1.0, 1.0): 0, (0.5, 0.5): 1, (0.5, 1.0): 2, (1.0, 0.5): 3,
              (1.0, 2.0): 4, (2.0, 1.0): 5, (2.0, 2.0): 6}


def mg5_id_map(lhe):
    """Parse <initrwgt> weight ids -> (muR, muF) from the MG5 header; return
    the column index (in file order) of every id in the FIRST weightgroup."""
    op = gzip.open if lhe.endswith(".gz") else open
    ids = []
    with op(lhe, "rt") as f:
        for line in f:
            if "<weight id=" in line:
                m = re.search(r"id='(\d+)'.*muR=([\d.E+]+)\s+muF=([\d.E+]+)",
                              line)
                if m:
                    ids.append((m.group(1), float(m.group(2)), float(m.group(3))))
            if "</initrwgt>" in line:
                break
    col = {}
    for i, (wid, mur, muf) in enumerate(ids):
        col.setdefault((round(mur, 3), round(muf, 3)), i)   # first group wins
    return col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lhe", help="MG5 LHE for the id map (mg5 mode)")
    ap.add_argument("--powheg", action="store_true")
    ap.add_argument("--old", help="previous npz to validate against")
    args = ap.parse_args()

    z = dict(np.load(args.parent, allow_pickle=True))
    lp = np.asarray(z["l_plus"], float)
    lm = np.asarray(z["l_minus"], float)
    w = np.asarray(z["weight"], float)
    ws = np.asarray(z["w_scale"], float)

    if args.powheg:
        cols = POWHEG_IDS
    else:
        if not args.lhe:
            raise SystemExit("--lhe required for the MG5 id map")
        cols = mg5_id_map(args.lhe)
    sel = [cols[p] for p in SEVEN]
    ratio = ws[:, sel] / np.where(np.abs(ws[:, [cols[(1.0, 1.0)]]]) > 0,
                                  ws[:, [cols[(1.0, 1.0)]]], np.inf)
    w_scale = w[:, None] * ratio

    def pt(v): return np.hypot(v[:, 0], v[:, 1])
    def eta(v):
        pm = np.sqrt((v[:, 0:3] ** 2).sum(1))
        return np.arctanh(np.clip(v[:, 2] / pm, -1 + 1e-12, 1 - 1e-12))
    mll = np.asarray(z["mll"], float)
    m = ((mll >= 66) & (mll <= 116) & (pt(lp) > 27) & (pt(lm) > 27)
         & (np.abs(eta(lp)) < 2.5) & (np.abs(eta(lm)) < 2.5))
    dphi = (np.arctan2(lp[:, 1], lp[:, 0]) - np.arctan2(lm[:, 1], lm[:, 0])
            + np.pi) % (2 * np.pi) - np.pi
    ph = (np.tan((np.pi - np.abs(dphi)) / 2)
          / np.cosh((eta(lm) - eta(lp)) / 2))
    out = dict(mll=mll[m], pT_ll=np.asarray(z["pT_ll"], float)[m],
               y_ll=np.asarray(z["y_ll"], float)[m], phistar=ph[m],
               pT_lead=np.maximum(pt(lp), pt(lm))[m], w=w[m],
               w_scale=w_scale[m],
               w_scale_labels=np.array([f"muR={a} muF={b}" for a, b in SEVEN]))
    print(f"fiducial: {m.sum():,} of {len(m):,}")

    if args.old and os.path.exists(args.old):
        old = dict(np.load(args.old))
        for k in ("mll", "pT_ll", "phistar", "pT_lead", "w"):
            if k not in old:
                continue
            same = (len(old[k]) == len(out[k])
                    and np.allclose(old[k], out[k], rtol=1e-10, atol=0))
            print(f"  vs old {k:8s}: "
                  + ("IDENTICAL" if same else
                     f"DIFFERS (old n={len(old[k])}, new n={len(out[k])})"))
    np.savez_compressed(args.out, **out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
