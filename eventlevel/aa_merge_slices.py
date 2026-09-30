#!/usr/bin/env python3
"""Combine several 1M-event runs of the SAME merged-diphoton slice into one
prior file (MG5 caps a run at 1M events, so a slice is produced in pieces).

Convention of the pieces (diphoton_shower_merged.py): per event `weight` is the
CKKW-L weight relative to that run's mean over all its generated events,
`sigma_pb` is that mean in pb (= sigma_LHE x <w_merge>), `n_generated` the
run's LHE events.  Pooling k runs of one slice means every event contributes
   sigma_LHE,f * w_merge,j / N_tot,           N_tot = sum_f N_f,
so the combined file carries n_generated = N_tot, sigma_pb = the N-weighted
mean of the pieces' sigma_pb, and each event's weight rescaled by
sigma_pb,f / sigma_pb so that the loaders' weight * sigma_pb / n_generated
reproduces exactly the expression above.  Concatenating the pieces as separate
slices would count the slice k times.

usage: aa_merge_slices.py OUT.npz PIECE1.npz PIECE2.npz ...
"""
import sys
import numpy as np


def main():
    out, pieces = sys.argv[1], sys.argv[2:]
    if len(pieces) < 1:
        sys.exit(__doc__)
    parts = [dict(np.load(f, allow_pickle=True)) for f in pieces]
    N = np.array([float(p["n_generated"]) for p in parts])
    S = np.array([float(p["sigma_pb"]) for p in parts])
    n_tot = N.sum(); sig = float((N * S).sum() / n_tot)
    keys = [k for k in parts[0] if isinstance(parts[0][k], np.ndarray) and parts[0][k].ndim >= 1
            and parts[0][k].shape[0] == len(parts[0]["weight"])]
    arrs = {}
    for k in keys:
        cols = []
        for p, s_f in zip(parts, S):
            a = np.asarray(p[k])
            if k == "weight":
                a = a.astype(float) * (s_f / sig)
            cols.append(a)
        arrs[k] = np.concatenate(cols)
    meta = {k: parts[0][k] for k in parts[0] if k not in keys and k not in ("n_generated", "sigma_pb", "n_vetoed")}
    nveto = sum(int(p.get("n_vetoed", 0)) for p in parts)
    np.savez_compressed(out, n_generated=int(n_tot), sigma_pb=sig, n_vetoed=nveto,
                        pieces=np.array(pieces), **meta, **arrs)
    fid = sig * n_tot and sig * arrs["weight"].sum() / n_tot
    print(f"{out}: {len(pieces)} pieces, {int(n_tot):,} generated, {len(arrs['weight']):,} selected, "
          f"sigma {sig:.4f} pb (pieces {np.round(S, 4)}), fiducial {fid:.4f} pb")


if __name__ == "__main__":
    main()
