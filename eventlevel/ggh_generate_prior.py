#!/usr/bin/env python3
"""gg->H LO+PS prior at 13 TeV: Pythia8 HiggsSM:gg2H, H kept stable, ISR/FSR on, MPI/hadronization off.
Stores (pT_H, y_H) + shower-variation weights + JET observables (anti-kT R=0.4
via SlowJet on the final state excluding the Higgs, pT>20, |eta|<4.5):
pT_j1, y_j1, njet20, dy_Hj1.  Jet arrays are 0/NaN when no jet passes.
Inclusive (no cuts), matching the NNLOJET H run."""
import argparse, numpy as np, pythia8
ap = argparse.ArgumentParser()
ap.add_argument("--nevt", type=int, default=3_000_000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", type=str, required=True)
args = ap.parse_args()
p = pythia8.Pythia()
p.readString("Beams:eCM = 13000")
p.readString("PDF:pSet = LHAPDF6:CT18NNLO/0")
p.readString("HiggsSM:gg2H = on")
p.readString("25:mayDecay = off")
p.readString("25:isVisible = off")   # SlowJet select=2 then never clusters the H
p.readString("25:m0 = 125.0")
p.readString("PartonLevel:ISR = on"); p.readString("PartonLevel:FSR = on")
p.readString("PartonLevel:MPI = off"); p.readString("HadronLevel:all = off")
p.readString('UncertaintyBands:doVariations = on')
p.readString('UncertaintyBands:List = {isrHi isr:muRfac=0.5, isrLo isr:muRfac=2.0, fsrHi fsr:muRfac=0.5, fsrLo fsr:muRfac=2.0}')
p.readString("Random:setSeed = on"); p.readString(f"Random:seed = {args.seed}")
p.readString("Print:quiet = on")
assert p.init()
out={k:[] for k in ('pT_H','y_H','weight','w_shower',
                    'pT_j1','y_j1','njet20','dy_Hj1')}
# anti-kT R=0.4 on the visible final state -- the H is flagged invisible
# above, so it is excluded at the source (no fragile proximity matching).
# pT>20 keeps room for later cuts; the power-counting seam for Q=m_H is 37.
sj = pythia8.SlowJet(-1, 0.4, 20.0, 4.5, 2, 2)
n=0
import time; t0=time.time()
while n < args.nevt:
    if not p.next(): continue
    n+=1
    for i in range(1, p.event.size()):
        q=p.event[i]
        if q.id()==25 and q.isFinal():
            out['pT_H'].append(q.pT()); out['y_H'].append(q.y())
            info=p.infoPython(); w=info.weight()
            out['weight'].append(w)
            out['w_shower'].append([info.weightValueByIndex(j)*w/max(info.weightValueByIndex(0),1e-300)
                                    for j in range(info.numberOfWeights())])
            sj.analyze(p.event)
            jets = [(sj.p(k).pT(), sj.p(k).rap()) for k in range(sj.sizeJet())]
            if jets:
                jets.sort(reverse=True)
                out['pT_j1'].append(jets[0][0]); out['y_j1'].append(jets[0][1])
                out['njet20'].append(len(jets))
                out['dy_Hj1'].append(q.y()-jets[0][1])
            else:
                out['pT_j1'].append(0.0); out['y_j1'].append(float('nan'))
                out['njet20'].append(0); out['dy_Hj1'].append(float('nan'))
            break
    if n % 250000 == 0:
        print(f"  {n:,} events  rate={n/max(time.time()-t0,1):.0f}/s", flush=True)
sigma_pb = p.infoPython().sigmaGen()*1e9
np.savez_compressed(args.out, sigma_pb=sigma_pb, n_generated=n,
                    **{k:np.asarray(v) for k,v in out.items()})
print(f"DONE: {n:,} events  sigma={sigma_pb:.3f} pb -> {args.out}")
