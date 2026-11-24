"""Check the same low-degree fits against each corresponding scale input."""
from pathlib import Path
import json
import sys
import numpy as np
BASE=Path('/Users/benassi/reweighting/zj_precision_20261115')
sys.path[:0]=[str(BASE/'code'),str(BASE/'code/examples')]
import zj_benchmark as bench
from maxent_upgrade.solver import _newton_maxent
root=Path(__file__).resolve().parent
data=BASE/'code/examples/data/zj_benchmark'
bench.DEFS=json.loads((data/'definitions.json').read_text())
e=bench.load_prior('/Users/benassi/reweighting/prior_v6/built/zj_prior_v6_nomec10M_par.npz')
p=e['weight']/e['weight'].sum()
H,_=bench.indicators(e)
mask=H[:,15:79].sum(0)>=50
cx=np.polynomial.chebyshev.chebvander(2*np.log(e['ptj1']/60)/np.log(1000/60)-1,4)
cy=np.polynomial.chebyshev.chebvander(2*np.log(e['ptj2']/60)/np.log(1000/60)-1,4)
train=bench.aggregate(np.load(data/'training_fo.npz'))
valid=bench.aggregate(np.load(data/'validation_fo.npz'))
rows=[]
for degree,joint in [(1,0),(4,0),(4,2)]:
    ix=[(k,0) for k in range(1,degree+1)]+[(0,k) for k in range(1,degree+1)]
    ix += [(i,j) for i in range(1,joint+1) for j in range(1,joint+1)]
    A=np.column_stack([cx[:,i]*cy[:,j] for i,j in ix])
    sd=np.sqrt(p@(A-p@A)**2+1e-30)
    for s in range(7):
        mu=train['moments'][:,:,s]/train['norm'][0,s]
        q,lam,ok=_newton_maxent(A,p,np.array([mu[i,j] for i,j in ix]),l2=1e-4)
        assert ok
        ref=valid['valid_xy'][:,:,s].ravel()/valid['norm'][0,s]
        pred=q@H[:,15:79]; prior=p@H[:,15:79]
        rows.append(dict(degree=degree,joint=joint,scale_index=s,
            prior_distance_supported=float(.5*abs(prior[mask]-ref[mask]).sum()),
            distance_supported=float(.5*abs(pred[mask]-ref[mask]).sum()),
            distance_full=float(.5*abs(pred-ref).sum()),coefficients=(lam/sd).tolist()))
(root/'scales.json').write_text(json.dumps(rows,indent=2)+'\n')
print('COMPLETE')
