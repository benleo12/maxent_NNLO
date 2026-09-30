#!/usr/bin/env python3
"""Shower a merged MG5 LO diphoton LHE file (pp -> aa @0, pp -> aa j @1,
Durham-kT cut ``ktdurham``) with Pythia 8 CKKW-L merging, apply the ATLAS
8 TeV fiducial selection of diphoton_generate_v5.py, and write the same npz
layout the diphoton scripts read.

Merging: Pythia's kT-merging (Merging:doKTMerging) with the merging scale set
to the MG5 ktdurham value, the two photons as the hard process (pp>aa), at
most one additional jet.  Merging:includeWeightInXsection puts the CKKW-L
weight into Info::weight(), so the per-event weight is XWGTUP times the
merging weight and vetoed events come back with weight zero.

Output weight convention (read by aa_eventlevel_solve.load_prior):
  weight      per-event RELATIVE weight, mean 1 over every generated LHE event
              (vetoed events included with 0)
  sigma_pb    mean of Info::weight() over the generated events, in pb: the
              merged cross section of this LHE file
  n_generated number of LHE events processed
so sigma_pb / n_generated * weight is the absolute weight of a selected event,
exactly as sigma_pb / n_generated * 1 is for the Pythia-internal slices.
"""
import argparse, math, os, sys, time
import numpy as np
import pythia8

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from diphoton_generate_v5 import frixione_ok  # noqa: E402  (same envelope, same ETmax)

ap = argparse.ArgumentParser()
ap.add_argument("--lhe", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--max-events", type=int, default=0)
ap.add_argument("--pdf", type=str, default="LHAPDF6:CT18NNLO/0", help="must match the LHE PDF")
ap.add_argument("--tms", type=float, default=20.0, help="merging scale = MG5 ktdurham")
ap.add_argument("--dparameter", type=float, default=0.4)
ap.add_argument("--njetmax", type=int, default=1)
ap.add_argument("--pt1min", type=float, default=40.0)
ap.add_argument("--pt2min", type=float, default=30.0)
ap.add_argument("--ymax", type=float, default=2.37)
ap.add_argument("--crack_lo", type=float, default=1.37)
ap.add_argument("--crack_hi", type=float, default=1.56)
ap.add_argument("--dRmin", type=float, default=0.4)
ap.add_argument("--riso", type=float, default=0.4)
ap.add_argument("--etmax", type=float, default=11.0)
ap.add_argument("--no-merging", action="store_true", help="debug: shower without CKKW-L")
args = ap.parse_args()

p = pythia8.Pythia()
p.readString("Beams:frameType = 4")
p.readString(f"Beams:LHEF = {args.lhe}")
p.readString(f"PDF:pSet = {args.pdf}")
if not args.no_merging:
    p.readString("Merging:doKTMerging = on")
    p.readString("Merging:ktType = 1")
    p.readString(f"Merging:Dparameter = {args.dparameter}")
    p.readString(f"Merging:nJetMax = {args.njetmax}")
    p.readString("Merging:Process = pp>aa")
    p.readString(f"Merging:TMS = {args.tms}")
    p.readString("Merging:includeWeightInXsection = on")
p.readString("PartonLevel:ISR = on"); p.readString("PartonLevel:FSR = on")
p.readString("PartonLevel:MPI = off"); p.readString("HadronLevel:all = off")
p.readString("Random:setSeed = on"); p.readString(f"Random:seed = {args.seed}")
p.readString("Print:quiet = on")
if not p.init():
    sys.exit("Pythia init failed")

out = {k: [] for k in ("g1", "g2", "m_aa", "pt_aa", "y_aa", "dphi_aa", "weight")}
ngen = 0; nsel = 0; nveto = 0; wsum = 0.0; wsum2 = 0.0
t0 = time.time()
while True:
    if not p.next():
        if p.infoPython().atEndOfFile():
            break
        continue
    ngen += 1
    info = p.infoPython()
    wev = info.weight()
    wsum += wev; wsum2 += wev * wev
    if args.max_events and ngen >= args.max_events:
        pass
    if wev == 0.0:
        nveto += 1
        if args.max_events and ngen >= args.max_events:
            break
        continue
    gs = []
    for i in range(1, p.event.size()):
        q = p.event[i]
        if q.isFinal() and q.id() == 22 and q.pT() > 20.0:
            gs.append((q.pT(), i))
    ok = len(gs) >= 2
    if ok:
        gs.sort(reverse=True)
        (pt1, i1), (pt2, i2) = gs[0], gs[1]
        g1, g2 = p.event[i1], p.event[i2]
        ok = pt1 >= args.pt1min and pt2 >= args.pt2min
    if ok:
        for g in (g1, g2):
            ay = abs(g.y())
            if ay > args.ymax or (args.crack_lo < ay < args.crack_hi):
                ok = False
    if ok:
        dy = g1.y() - g2.y(); dp = abs(g1.phi() - g2.phi())
        if dp > math.pi:
            dp = 2 * math.pi - dp
        ok = math.hypot(dy, dp) >= args.dRmin
    if ok:
        ok = frixione_ok(p, i1, args.riso, args.etmax) and frixione_ok(p, i2, args.riso, args.etmax)
    if ok:
        px = g1.px() + g2.px(); py = g1.py() + g2.py(); pz = g1.pz() + g2.pz(); E = g1.e() + g2.e()
        m2 = E * E - px * px - py * py - pz * pz
        out["g1"].append((g1.px(), g1.py(), g1.pz(), g1.e()))
        out["g2"].append((g2.px(), g2.py(), g2.pz(), g2.e()))
        out["m_aa"].append(math.sqrt(max(m2, 0.0)))
        out["pt_aa"].append(math.hypot(px, py))
        out["y_aa"].append(0.5 * math.log((E + pz) / max(E - pz, 1e-30)))
        out["dphi_aa"].append(dp)
        out["weight"].append(wev)
        nsel += 1
        if nsel % 100000 == 0:
            print(f"  gen={ngen:,} sel={nsel:,} veto={nveto:,} rate={ngen/max(time.time()-t0,1):.0f} ev/s", flush=True)
    if args.max_events and ngen >= args.max_events:
        break

wmean = wsum / max(ngen, 1)
sig_lhe = sum(info.sigmaLHEF(i) for i in range(info.nProcessesLHEF()))
arrs = {k: np.asarray(v) for k, v in out.items() if len(v)}
arrs["weight"] = arrs["weight"] / wmean            # relative, mean 1 over generated
np.savez_compressed(args.out, sigma_pb=wmean, n_generated=ngen, n_vetoed=nveto,
                    sigma_lhe_pb=sig_lhe, sigma_gen_pythia_pb=info.sigmaGen() * 1e9,
                    tms=args.tms, channels="mg5 aa+aaj ckkwl", pthatmin=-1.0, pthatmax=-1.0, **arrs)
neff = wsum * wsum / max(wsum2, 1e-300)
print(f"DONE: {nsel:,} selected / {ngen:,} generated ({nveto:,} merging-vetoed)  "
      f"<w>={wmean:.4f} pb  sigma_LHE={sig_lhe:.4f} pb  sigmaGen(Pythia)={info.sigmaGen()*1e9:.4f} pb  "
      f"Neff/N={neff/max(ngen,1):.3f}  -> {args.out}")
