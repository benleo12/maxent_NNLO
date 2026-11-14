"""Small scientific implementation checks; no validation reference is read."""
import json
import numpy as np
from train import load_prior,make_features,cheb,umap,profile_w,_newton_maxent,MODELS,L2
from aggregate import training_tags

for charge in ['WP','WM']:
    ev,p,meta=load_prior(charge)
    labels=training_tags();idx={x:i for i,x in enumerate(labels)};r=np.zeros((len(idx),7));r[0]=1
    for pre,cs,w in [
        ('prof_mw_',cheb(umap(ev['m_w'],40,1000,('bw',80.385,2.085)),12),np.ones(len(p))),
        ('prof_absyw_',cheb(umap(ev['y_abs'],0,4,'lin'),12),np.ones(len(p))),
        ('prof_wptw_',cheb(umap(ev['pt_w'],25,500,'log'),20),profile_w(ev['pt_w'],25,50,500)),
        ('prof_wctw_',cheb(ev['ctw'],6),profile_w(ev['ctw']+1,.3,.4,1.6,1.7))]:
        for n in range(cs.shape[1]):
            if pre+str(n) in idx:r[idx[pre+str(n)]]=float((p*w)@cs[:,n])
    cy=cheb(umap(ev['y_abs'],0,4,'lin'),4);cc=cheb(ev['ctw'],4);w=profile_w(ev['ctw']+1,.3,.4,1.6,1.7)
    for k in range(1,5):
        for n in range(5):r[idx['prof_yc_%d%d'%(k,n)]]=float((p*w)@(cy[:,k]*cc[:,n]))
    t=dict(idx=idx,r=r);order={'m_w':12,'y_abs':12,'pt_w':20}
    for model in MODELS:
        phi,b,L,names,pdep,diag,S=make_features(ev,p,t,order,model)
        target=b+L@r[:,0];err=float(np.max(np.abs(p@phi-target)))
        assert err<5e-10,(charge,model,err)
        q,lam,ok=_newton_maxent(phi,p,target,l2=L2)
        assert ok and np.max(np.abs(q/p-1))<1e-9
        assert np.max(np.abs(p@diag-S@r[:,0]))<5e-10
        print(json.dumps(dict(charge=charge,model=model,prior_equal_target_max_error=err,n_fid=len(p))),flush=True)
        del phi,pdep,diag
