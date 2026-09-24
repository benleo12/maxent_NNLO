#!/usr/bin/env python3
r"""Build the ATLAS-fiducial npz from a showered parent that stores BORN-level
leptons (l_plus_born / l_minus_born: final lepton + its own QED FSR photons).

Same frozen recut and phistar recipe as make_dy_atlas_npz.py, but applied to
the BORN leptons, which is the convention of the Born-level ATLAS tables and
of a QCD-only fixed-order calculation.  The bare-lepton quantities are kept
alongside under *_bare so nothing is lost.

    66 <= mll <= 116,  pT(l+-) > 27,  |eta(l+-)| < 2.5   (on BORN leptons)
    phistar = tan((pi-|dphi|)/2) / cosh((eta- - eta+)/2)

Scale weights (generators): w_scale[:, k] = w * rwgt_k / rwgt_0 over the
7-point set, id map parsed from the MG5 LHE header or the POWHEG chain map.
Shower weights (prior): w_shower passed through.

Usage:
  make_dy_atlas_npz_born.py --parent dy_psLO_born_1.npz --out dy_prior_atlas_v3.npz
  make_dy_atlas_npz_born.py --parent dy_mcatnlo_born_sh.npz --lhe dy_NLO_500k.lhe.gz --out dy_mcatnlo_atlas_v3.npz
  make_dy_atlas_npz_born.py --parent A.npz --parent B.npz --powheg --out dy_minnlo_atlas_v4.npz
"""
import argparse, os, numpy as np
from make_dy_atlas_npz import SEVEN, POWHEG_IDS, mg5_id_map
import re

def powheg_id_map(lhe):
    """POWHEG <initrwgt> -> {(muR, muF): column}.  Columns are indexed by the
    FIRST appearance of each weight id (a header may declare an id twice, once
    as 'scale variation NNNN' and once with its muR/muF description); the
    (muR, muF) of an id is taken from whichever line carries it, in either the
    'renscfact=.. facscfact=..' or the 'muR=.. muF=..' form, 'd0' suffixes
    allowed.  Every column is w * rwgt_k / rwgt_0 downstream, so which of two
    equivalent ids is chosen is immaterial for the envelope."""
    col_of_id = {}; key_of_id = {}; inside = False
    with open(lhe) as f:
        for line in f:
            if "<initrwgt>" in line: inside = True; continue
            if "</initrwgt>" in line: break
            if not (inside and "<weight " in line): continue
            mid = re.search(r"id=['\"]([^'\"]+)['\"]", line)
            if not mid: continue
            wid = mid.group(1)
            if wid not in col_of_id: col_of_id[wid] = len(col_of_id)
            m = (re.search(r"renscfact=([\d.]+)d?0?\s+facscfact=([\d.]+)d?0?", line)
                 or re.search(r"muR=([\d.]+)d?0?\s+muF=([\d.]+)d?0?", line))
            if m:
                key_of_id.setdefault(wid, (round(float(m.group(1)), 3), round(float(m.group(2)), 3)))
    col = {}
    for wid, key in key_of_id.items(): col.setdefault(key, col_of_id[wid])
    return col

def kin(lp, lm):
    def pt(v): return np.hypot(v[:, 0], v[:, 1])
    def eta(v):
        pm = np.sqrt((v[:, 0:3] ** 2).sum(1)); return np.arctanh(np.clip(v[:, 2] / np.maximum(pm, 1e-30), -1 + 1e-12, 1 - 1e-12))
    s = lp + lm
    mll = np.sqrt(np.maximum(s[:, 3] ** 2 - (s[:, 0:3] ** 2).sum(1), 0.0))
    pT_ll = np.hypot(s[:, 0], s[:, 1])
    y_ll = 0.5 * np.log((s[:, 3] + s[:, 2]) / np.maximum(s[:, 3] - s[:, 2], 1e-30))
    dphi = (np.arctan2(lp[:, 1], lp[:, 0]) - np.arctan2(lm[:, 1], lm[:, 0]) + np.pi) % (2 * np.pi) - np.pi
    ph = np.tan((np.pi - np.abs(dphi)) / 2) / np.cosh((eta(lm) - eta(lp)) / 2)
    fid = ((mll >= 66) & (mll <= 116) & (pt(lp) > 27) & (pt(lm) > 27)
           & (np.abs(eta(lp)) < 2.5) & (np.abs(eta(lm)) < 2.5))
    return dict(mll=mll, pT_ll=pT_ll, y_ll=y_ll, phistar=ph, pT_lead=np.maximum(pt(lp), pt(lm)),
                pT_sub=np.minimum(pt(lp), pt(lm))), fid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lhe", action="append", help="one per --parent, same order")
    ap.add_argument("--powheg", action="store_true")
    a = ap.parse_args()
    parts = [dict(np.load(f, allow_pickle=True)) for f in a.parent]
    cat = lambda k: np.concatenate([np.asarray(p[k], float) for p in parts])
    lpb, lmb, lp, lm, w = cat("l_plus_born"), cat("l_minus_born"), cat("l_plus"), cat("l_minus"), cat("weight")
    born, fid = kin(lpb, lmb)
    bare, fid_bare = kin(lp, lm)
    good = (lpb[:, 3] > 0) & (lp[:, 3] > 0)
    m = fid & good
    out = {k: v[m] for k, v in born.items()}
    out.update({k + "_bare": v[m] for k, v in bare.items()})
    out["in_fid_bare"] = fid_bare[m]
    out["w"] = w[m]
    # weights
    if all("w_scale" in p and np.size(p["w_scale"]) for p in parts):
        assert a.lhe and len(a.lhe) == len(parts), "give one --lhe per --parent"
        ratios = []
        for p, lhe in zip(parts, a.lhe):
            ws = np.asarray(p["w_scale"], float)
            cols = powheg_id_map(lhe) if a.powheg else mg5_id_map(lhe)
            wev = np.asarray(p["weight"], float)
            # A file that stores only the six variations (POWHEG rwl chain,
            # ids 1002-1007) has the central weight as the event weight itself.
            c0col = ws[:, cols[(1.0, 1.0)]] if (1.0, 1.0) in cols else wev
            den = np.where(np.abs(c0col) > 0, c0col, np.inf)[:, None]
            r = np.stack([(ws[:, cols[p_]] if p_ in cols else c0col) for p_ in SEVEN], 1) / den
            ratios.append(r)
            print(f"  {os.path.basename(lhe)}: {ws.shape[1]} weight columns; central {'stored' if (1.0,1.0) in cols else 'is the event weight'}; map {cols}")
        out["w_scale"] = (w[:, None] * np.concatenate(ratios))[m]
        out["w_scale_labels"] = np.array([f"muR={x} muF={y}" for x, y in SEVEN])
    if all("w_shower" in p and np.size(p["w_shower"]) for p in parts):
        out["w_shower"] = cat("w_shower")[m]
    np.savez_compressed(a.out, **out)
    print(f"{a.out}: Born-fiducial {m.sum():,} of {len(m):,} ({100*m.mean():.1f}%); "
          f"bare-fiducial would be {int((fid_bare&good).sum()):,}; "
          f"mean m_ll Born {np.average(out['mll'], weights=out['w']):.2f} bare {np.average(out['mll_bare'], weights=out['w']):.2f}; "
          f"keys {sorted(out)}")

if __name__ == "__main__":
    main()
