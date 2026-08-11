#!/usr/bin/env python3
"""Diphoton LO+PS prior v2 (James-spec cuts, 8 TeV): full kinematics + shower-variation weights.
Fixes the v1 acceptance catastrophe (pTHatMin=0 -> 3e-4 acceptance) with pTHatMin=25.
Selection identical to generate_prior.py: pT1>40, pT2>30, |y|<2.37 minus crack [1.37,1.56],
dR(aa)>0.4, Frixione-envelope isolation (linear in r, ETmax at riso), m_aa>80 implicit via CSV use.
"""
import argparse, math, numpy as np, pythia8

def frixione_ok(p, ig, riso, etmax):
    """Linear-in-r Frixione envelope with fixed ETmax (James spec)."""
    g = p.event[ig]
    for r_i in range(1, p.event.size()):
        pass
    # collect hadronic ET in cones
    ets = []
    for i in range(1, p.event.size()):
        q = p.event[i]
        if not q.isFinal() or i == ig: continue
        if q.id() == 22 and q.pT() < 1.0: continue
        dy = q.y() - g.y(); dp = abs(q.phi() - g.phi())
        if dp > math.pi: dp = 2*math.pi - dp
        dr = math.hypot(dy, dp)
        if dr < riso: ets.append((dr, q.pT()))
    ets.sort()
    cum = 0.0
    for dr, et in ets:
        cum += et
        if cum > etmax * (dr / riso):  # linear envelope
            return False
    return True

ap = argparse.ArgumentParser()
ap.add_argument("--nevt", type=int, default=10_000_000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--sqrtS", type=float, default=8000.0)
ap.add_argument("--pdf", type=str, default="LHAPDF6:CT18NNLO/0")
ap.add_argument("--pthatmin", type=float, default=25.0)
ap.add_argument("--pthatmax", type=float, default=-1.0,
                help="EXCLUSIVE upper pTHat bound. Required for slicing: with\n                      pTHatMin alone the slices overlap and concatenating them\n                      with sigma_i/n_i triple-counts the hard region.")
ap.add_argument("--pt1min", type=float, default=40.0)
ap.add_argument("--pt2min", type=float, default=30.0)
ap.add_argument("--ymax", type=float, default=2.37)
ap.add_argument("--crack_lo", type=float, default=1.37)
ap.add_argument("--crack_hi", type=float, default=1.56)
ap.add_argument("--dRmin", type=float, default=0.4)
ap.add_argument("--riso", type=float, default=0.4)
ap.add_argument("--etmax", type=float, default=11.0)
ap.add_argument("--shower_vars", action="store_true")
ap.add_argument("--out", type=str, required=True)
args = ap.parse_args()

p = pythia8.Pythia()
p.readString(f"Beams:eCM = {args.sqrtS}")
p.readString("Beams:idA = 2212"); p.readString("Beams:idB = 2212")
p.readString(f"PDF:pSet = {args.pdf}")
p.readString("PromptPhoton:all = off")
p.readString("PromptPhoton:ffbar2gammagamma = on")
p.readString("PromptPhoton:gg2gammagamma = on")  # box loop, dominant hard channel
p.readString(f"PhaseSpace:pTHatMin = {args.pthatmin}")
if args.pthatmax > 0:
    p.readString(f"PhaseSpace:pTHatMax = {args.pthatmax}")
p.readString("PartonLevel:ISR = on"); p.readString("PartonLevel:FSR = on")
p.readString("PartonLevel:MPI = off"); p.readString("HadronLevel:all = off")
if args.shower_vars:
    p.readString('UncertaintyBands:doVariations = on')
    p.readString('UncertaintyBands:List = {isrHi isr:muRfac=0.5, isrLo isr:muRfac=2.0, fsrHi fsr:muRfac=0.5, fsrLo fsr:muRfac=2.0}')
p.readString("Random:setSeed = on"); p.readString(f"Random:seed = {args.seed}")
p.readString("Print:quiet = on")
assert p.init()

out={k:[] for k in ('g1','g2','m_aa','pt_aa','y_aa','dphi_aa','weight','w_shower')}
ngen=0; nsel=0
import time; t0=time.time()
while ngen < args.nevt:
    if not p.next(): continue
    ngen += 1
    info=p.infoPython()
    wev=info.weight()
    # photons: two hardest final-state photons
    gs=[]
    for i in range(1, p.event.size()):
        q=p.event[i]
        if q.isFinal() and q.id()==22 and q.pT()>20.0:
            gs.append((q.pT(), i))
    if len(gs)<2: continue
    gs.sort(reverse=True)
    (pt1,i1),(pt2,i2)=gs[0],gs[1]
    g1,g2=p.event[i1],p.event[i2]
    if pt1<args.pt1min or pt2<args.pt2min: continue
    ok=True
    for g in (g1,g2):
        ay=abs(g.y())
        if ay>args.ymax or (args.crack_lo<ay<args.crack_hi): ok=False
    if not ok: continue
    dy=g1.y()-g2.y(); dp=abs(g1.phi()-g2.phi())
    if dp>math.pi: dp=2*math.pi-dp
    if math.hypot(dy,dp)<args.dRmin: continue
    if not (frixione_ok(p,i1,args.riso,args.etmax) and frixione_ok(p,i2,args.riso,args.etmax)): continue
    px=g1.px()+g2.px(); py=g1.py()+g2.py(); pz=g1.pz()+g2.pz(); E=g1.e()+g2.e()
    m2=E*E-px*px-py*py-pz*pz
    out['g1'].append((g1.px(),g1.py(),g1.pz(),g1.e()))
    out['g2'].append((g2.px(),g2.py(),g2.pz(),g2.e()))
    out['m_aa'].append(math.sqrt(max(m2,0.0)))
    out['pt_aa'].append(math.hypot(px,py))
    out['y_aa'].append(0.5*math.log((E+pz)/max(E-pz,1e-30)))
    out['dphi_aa'].append(dp)
    out['weight'].append(wev)
    if args.shower_vars:
        out['w_shower'].append([info.weightValueByIndex(i)*wev/max(info.weightValueByIndex(0),1e-300)
                                for i in range(info.numberOfWeights())])
    nsel+=1
    if nsel % 100000 == 0:
        print(f"  gen={ngen:,} sel={nsel:,} ({nsel/max(ngen,1)*100:.1f}%) rate={ngen/max(time.time()-t0,1):.0f} gen/s", flush=True)

sigma_pb = p.infoPython().sigmaGen()*1e9  # mb -> pb
arrs={k:np.asarray(v) for k,v in out.items() if len(v)}
np.savez_compressed(args.out, sigma_pb=sigma_pb, n_generated=ngen,
                    pthatmin=args.pthatmin, pthatmax=args.pthatmax, **arrs)
print(f"DONE: {nsel:,} selected / {ngen:,} generated  sigma_gen={sigma_pb:.3f} pb  -> {args.out}")
