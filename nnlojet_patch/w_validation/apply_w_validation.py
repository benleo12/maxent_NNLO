"""Add W prediction-study observables after the public NNLOJET v1.0.2 patch."""
from pathlib import Path
import argparse,hashlib,json
ap=argparse.ArgumentParser();ap.add_argument('nnlojet_source',type=Path);a=ap.parse_args()
core=a.nnlojet_source/'driver/core';here=Path(__file__).resolve().parent
ef=core/'EvalFuncs.f90';ob=core/'Observables.f90';e=ef.read_text();o=ob.read_text()
if 'function eval_wval_yctw' in e or '"wval_yc_11"' in o:
    raise SystemExit('W validation observables already present or partially installed; use a fresh source copy.')
for token in ['function eval_w_ptw','function eval_w_ctw','function eval_costhCS_w','function cheb_T']:
    assert token in e,'Apply the public baseline moment patch first: missing '+token
assert e.count('end module EvalFuncs_mod')==1
e=e.replace('end module EvalFuncs_mod',(here/'w_validation_functions.f90').read_text()+'\nend module EvalFuncs_mod')
for ilep in [1,2]:
    marker='      call bind_obs("costh_w", "signed cos th*_CS of the charged lepton (W rest frame)", EvalObs_t(eval_mem=eval_costhCS_w, imem=-%d))'%ilep
    assert o.count(marker)==1,'Expected exactly one charge-specific costh_w binding'
    extra=[]
    for n in range(1,21):
        extra.append('      call bind_obs("wval_pt_%d", "W recoil moment on acting range 25-500", &\n                    EvalObs_t(eval_mem=eval_wval_pt, imem=-%d))'%(n,n))
    for k in range(1,5):
        for n in range(5):
            extra.append('      call bind_obs("wval_yc_%d%d", "windowed W rapidity decay product %d%d", &\n                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-%d))'%(k,n,k,n,ilep*1000+k*10+n))
    for cut in [27,30]:
        extra.append('      call bind_obs("wval_eta%d", "charged-lepton eta after pT>%d", &\n                    EvalObs_t(eval_mem=eval_wval_eta, imem=-%d))'%(cut,cut,ilep*100+cut))
    o=o.replace(marker,marker+'\n'+'\n'.join(extra))
before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ef,ob]}
ef.write_text(e);ob.write_text(o)
after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ef,ob]}
print(json.dumps(dict(before=before,after=after),indent=2))
