"""Finite-difference check of the training-residual response, before unblinding.

Use a fixed 6,000-event subset and artificial, nontrivial targets built from it.
No production fixed-order input or held-out distribution is read.
"""
import json
import numpy as np
from train import (ROOT,load_prior,make_features,linear_residual_cov,cheb,umap,
                   profile_w,_newton_maxent,L2)
from aggregate import training_tags

ev,p,meta=load_prior('WP')
ix=np.linspace(0,len(p)-1,6000,dtype=int);ev={k:v[ix] for k,v in ev.items()};p=p[ix];p/=p.sum()
labels=training_tags();idx={x:i for i,x in enumerate(labels)}
cy=cheb(umap(ev['y_abs'],0,4,'lin'),12);cc=cheb(ev['ctw'],6)
wp=profile_w(ev['pt_w'],25,50,500);wc=profile_w(ev['ctw']+1,.3,.4,1.6,1.7)
truth=p*np.exp(.16*cy[:,1]-.12*cc[:,1]+.08*cy[:,1]*cc[:,1]+.05*wp);truth/=truth.sum()
r=np.zeros((len(idx),7));r[0]=1
for pre,cs,w in [
    ('prof_mw_',cheb(umap(ev['m_w'],40,1000,('bw',80.385,2.085)),12),np.ones(len(p))),
    ('prof_absyw_',cy,np.ones(len(p))),
    ('prof_wptw_',cheb(umap(ev['pt_w'],25,500,'log'),20),wp),
    ('prof_wctw_',cc,wc)]:
    for n in range(cs.shape[1]):
        if pre+str(n) in idx:r[idx[pre+str(n)]]=float((truth*w)@cs[:,n])
for k in range(1,5):
    for n in range(5):r[idx['prof_yc_%d%d'%(k,n)]]=float((truth*wc)@(cy[:,k]*cc[:,n]))
t=dict(idx=idx,r=r,cr=np.eye(len(r))*1e-8)
orders={'m_w':3,'y_abs':4,'pt_w':5};records=[]
for model in ['A','J2']:
    phi,b,L,names,pdep,H,S=make_features(ev,p,t,orders,model)
    q,lam,ok=_newton_maxent(phi,p,b+L@r[:,0],l2=L2);assert ok
    cov,cf,cp,response,hess=linear_residual_cov(phi,p,q,pdep,H,L,S,t['cr'],lam)
    D=response@L-S
    def compare(kind,direction,epsilon,actual,predicted):
        scale=max(float(np.max(np.abs(predicted))),1e-12)
        error=float(np.max(np.abs(actual-predicted))/scale)
        row=dict(model=model,kind=kind,direction=direction,epsilon=epsilon,
                 relative_max_norm_error=error,max_abs_error=float(np.max(np.abs(actual-predicted))),
                 predicted_max_abs=scale)
        records.append(row);print(json.dumps(row),flush=True)
        assert error<2e-3,row
    for tag in ['prof_wptw_0','prof_yc_10','prof_yc_21']:
        direction=np.zeros(len(r));direction[idx[tag]]=1
        for epsilon in [2e-5,1e-5]:
            residual=[]
            for sign in [-1,1]:
                rp=r[:,0]+sign*epsilon*direction
                qp,lp,ok=_newton_maxent(phi,p,b+L@rp,l2=L2,lam0=lam);assert ok
                residual.append(qp@H-S@rp)
            compare('target',tag,epsilon,(residual[1]-residual[0])/(2*epsilon),D@direction)
    f=q@phi;h=q@H;mp=p@phi;sd2=p@(phi-mp)**2
    dprior=pdep-L2*((phi-mp)**2-sd2)*(lam/np.sqrt(sd2))
    influence=q[:,None]*((H-h)-(phi-f)@response.T)+p[:,None]*(dprior@response.T)
    for name,score in [('recoil',wp+.3*cy[:,2]),('rapidity_angle',cy[:,1]*cc[:,1]+.2*wc)]:
        score=score-p@score
        for epsilon in [2e-4,1e-4]:
            residual=[]
            for sign in [-1,1]:
                pp=p*np.exp(sign*epsilon*score);pp/=pp.sum()
                phip,bp,Lp,np_,dp,Hp,Sp=make_features(ev,pp,t,orders,model)
                assert np.array_equal(phip,phi) and np.array_equal(Hp,H)
                qp,lp,ok=_newton_maxent(phip,pp,bp+Lp@r[:,0],l2=L2);assert ok
                residual.append(qp@Hp-Sp@r[:,0])
            compare('prior',name,epsilon,(residual[1]-residual[0])/(2*epsilon),score@influence)
out=dict(dataset='Deterministic 6000-event subset; synthetic target tilt; no production targets or validation predictions',
         fixed_orders=orders,ridge=L2,checks=records,passed=True,
         approximation='First-order response about a fixed constraint set. Statistical covariance estimated from finite independent runs; the chi-square selection diagnostic is not a coverage claim.')
(ROOT/'output/JACOBIAN_CHECK.json').write_text(json.dumps(out,indent=2)+'\n')
