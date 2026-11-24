"""Follow-up diagnostics of the real Z+2-jet correction. Run on a compute host.

Fits use the original training moments only. The reference has been inspected
previously, so this is a follow-up analysis, not a new blinded validation.
All specified degrees and both common-support and full-grid errors are saved.
No manuscript source is modified.
"""
from pathlib import Path
import json
import sys
import hashlib
import warnings
import numpy as np

for msg in ('divide by zero encountered in matmul', 'overflow encountered in matmul',
            'invalid value encountered in matmul'):
    warnings.filterwarnings('ignore', message=msg)
BASE = Path('/Users/benassi/reweighting/zj_precision_20261115')
sys.path[:0] = [str(BASE/'code'), str(BASE/'code/examples')]
import zj_benchmark as bench
from maxent_upgrade.solver import _newton_maxent

OUT = Path(__file__).resolve().parent
DATA = BASE/'code/examples/data/zj_benchmark'
bench.DEFS = json.loads((DATA/'definitions.json').read_text())
prior_path = Path('/Users/benassi/reweighting/prior_v6/built/zj_prior_v6_nomec10M_par.npz')
events = bench.load_prior(prior_path)
p = events['weight']/events['weight'].sum()
H, _ = bench.indicators(events)
grid = H[:,15:79]
counts = grid.sum(0)
mask = counts >= 50
train = np.load(DATA/'training_fo.npz')
valid = np.load(DATA/'validation_fo.npz')
target = bench.aggregate(train)
mu = target['moments'][:,:,0]/target['norm'][0,0]
vr = bench.fo_vector(valid)
ref = bench.transformed(vr.mean(0)/valid['norm'][:,0,:].mean(0)[:,None])[0]
u = 2*np.log(events['ptj1']/60)/np.log(1000/60)-1
v = 2*np.log(events['ptj2']/60)/np.log(1000/60)-1
cx = np.polynomial.chebyshev.chebvander(u,4)
cy = np.polynomial.chebyshev.chebvander(v,4)
models = [('prior', [])]
models += [(f'D{d}_marginal', [(k,0) for k in range(1,d+1)]+[(0,k) for k in range(1,d+1)]) for d in range(1,5)]
models += [(f'D{d}_joint', [(i,j) for i in range(d+1) for j in range(d+1) if i+j]) for d in range(1,5)]
models += [('D4_J2', [(k,0) for k in range(1,5)]+[(0,k) for k in range(1,5)]+[(i,j) for i in (1,2) for j in (1,2)])]
protocol = dict(task='Inspect the measured QCD correction, not a constructed target',
    status='Follow-up using previously inspected independent reference',
    prior=str(prior_path), models=[dict(name=n,indices=i) for n,i in models],
    support_rule='Original validation cells with >=50 prior events; also retain full-grid metrics',
    replicas=128, seed=18112026,
    metrics=['full and supported 2D distance','supported conditional log-correction RMS',
             'supported conditional KL divergence','conditional jet-balance probabilities'],
    interpretation='LO Z+2jet hard correction to LO DY shower; no general NNLOPS or high-dimensional claim')
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')

def metrics(pred, reference):
    a,b=pred[15:79],reference[15:79]
    ac,bc=a[mask]/a[mask].sum(),b[mask]/b[mask].sum()
    return np.array([.5*abs(a-b).sum(),.5*abs(a[mask]-b[mask]).sum(),
        np.sqrt(bc @ np.log(ac/bc)**2), bc @ np.log(bc/ac)])

preds=[]; coeff=[]; weights=[]; mats=[]; metrics_c=[]; closures=[]
for name,indices in models:
    if indices:
        A=np.column_stack([cx[:,i]*cy[:,j] for i,j in indices])
        t=np.array([mu[i,j] for i,j in indices])
        q,lam,ok=_newton_maxent(A,p,t,l2=1e-4)
        assert ok
        sd=np.sqrt(p @ (A-p@A)**2+1e-30)
        c=lam/sd
        closures.append(float(abs(q@A-t).max()))
    else:
        A=np.empty((len(p),0)); q=p; c=np.empty(0); closures.append(0.)
    pred=bench.transformed(q@H)
    preds.append(pred); coeff.append(c); weights.append(q); mats.append(A)
    metrics_c.append(metrics(pred,ref))
    print(name,'K',len(indices),'metrics',metrics_c[-1].tolist(),flush=True)

rng=np.random.default_rng(18112026)
boots=[]; refboots=[]; metricboots=[]; coefboots=[[] for _ in models]
for r in range(128):
    pr=p*rng.poisson(1,len(p)); pr/=pr.sum()
    ti=rng.integers(len(train['seeds']),size=len(train['seeds']))
    t=bench.aggregate(train,ti); m=t['moments'][:,:,0]/t['norm'][0,0]
    vi=rng.integers(len(valid['seeds']),size=len(valid['seeds']))
    rf=bench.transformed(vr[vi].mean(0)/valid['norm'][vi,0,:].mean(0)[:,None])[0]
    rp=[]; rm=[]
    for j,(_,indices) in enumerate(models):
        if indices:
            A=mats[j]; keep=pr>0
            tar=np.array([m[i,k] for i,k in indices])
            q,lam,ok=_newton_maxent(A[keep],pr[keep],tar,l2=1e-4)
            assert ok, (r,j)
            pred=bench.transformed(q@H[keep])
            sd=np.sqrt(pr @ (A-pr@A)**2+1e-30)
            coefboots[j].append(lam/sd)
        else:
            pred=bench.transformed(pr@H); coefboots[j].append([])
        rp.append(pred); rm.append(metrics(pred,rf))
    boots.append(rp); refboots.append(rf); metricboots.append(rm)
    if (r+1)%16==0: print('replicas',r+1,flush=True)
boots=np.array(boots); metricboots=np.array(metricboots); preds=np.array(preds)
weights=np.array(weights); metrics_c=np.array(metrics_c)
pgrid=p@grid; fgrid=ref[15:79]
ratio=np.divide(fgrid,pgrid,out=np.full(64,np.nan),where=pgrid>0)
summary=dict(protocol=protocol,n_prior=len(p),covered_reference_mass=float(fgrid[mask].sum()),
    prior_input_weight_sum=float(events['weight'].sum()),fo_input_norm_mean=float(target['norm'][0,0]),
    normalization='Shape-only analysis. The two raw input normalization sums have different conventions and are not compared as physical rates.',
    models=[dict(name=n,constraints=len(i),indices=i,coefficients=c.tolist(),
        coefficient_sd=np.std(coefboots[j],axis=0,ddof=1).tolist(),
        coefficient_covariance=(np.cov(np.asarray(coefboots[j]).T).tolist() if i else []),
        metrics=metrics_c[j].tolist(),metrics_sd=metricboots[:,j].std(0,ddof=1).tolist(),
        closure_abs=closures[j]) for j,((n,i),c) in enumerate(zip(models,coeff))],
    grid=[dict(i=int(k//8),j=int(k%8),count=int(counts[k]),
        prior=float(pgrid[k]),reference=float(fgrid[k]),ratio=float(ratio[k]),
        fitted_ratio=float((weights[-1]@grid)[k]/pgrid[k])) for k in np.flatnonzero(mask)],
    balance=dict(edges=bench.DEFS['balance_edges'],intervals=bench.DEFS['balance_intervals'],
        prior=preds[0,3:15].reshape(2,6).tolist(),reference=ref[3:15].reshape(2,6).tolist(),
        selected=preds[-1,3:15].reshape(2,6).tolist()))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
np.savez_compressed(OUT/'diagnostics.npz', methods=np.array([x[0] for x in models]),
    predictions=preds,reference=ref,bootstrap=boots,reference_bootstrap=refboots,
    metrics=metrics_c,metrics_bootstrap=metricboots,grid_counts=counts,common_mask=mask,
    prior_grid=pgrid,target_grid=fgrid,correction_ratio=ratio,central_grid=weights@grid,
    u=u,v=v,prior_weights=p,central_weights=weights)
print('COMPLETE',flush=True)
