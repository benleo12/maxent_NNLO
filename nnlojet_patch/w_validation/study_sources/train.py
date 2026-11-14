"""Training-only W study. This program never opens a validation accumulator.

The existing package supplies the maps, windows and Newton solve. Targets are
linear functions of the common vector of normalized fixed-order accumulators,
so their full Monte Carlo covariance is retained.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'package'))
from maxent_upgrade.basis import cheb,umap
from maxent_upgrade.constraints import profile_w
from maxent_upgrade.solver import _newton_maxent

L2=1.e-4
MODELS=['B','A','J2','J4']

def chi2_sf(x,rank):
    """Chi-square survival for integer rank, by the incomplete-gamma recurrence."""
    t=np.asarray(x,float)/2.
    if rank%2:
        q=np.vectorize(math.erfc)(np.sqrt(t));a=.5
    else:
        q=np.exp(-t);a=1.
    while a<rank/2:
        with np.errstate(divide='ignore',invalid='ignore'):
            term=np.exp(a*np.log(t)-t-math.lgamma(a+1))
        q=q+np.where(t==0,0,term);a+=1.
    return np.clip(q,0,1)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_prior(charge):
    fn=Path('/Users/benassi/reweighting/prior_v6/built/w_prior_v6_nomec_%s.npz'%charge.lower())
    z=np.load(str(fn),allow_pickle=False)
    fid=(z['pt_l']>20)&(np.abs(z['y_l'])<2.5)&(z['pt_nu']>25)&(z['mt']>40)
    fid &= (z['weight']>0)&np.isfinite(z['m_w'])&(z['charge']==(1 if charge=='WP' else -1))
    l=z['l_born'][fid];nu=z['nu'][fid];v=l+nu
    m=np.sqrt(np.maximum(v[:,3]**2-(v[:,:3]**2).sum(1),0));qt=np.hypot(v[:,0],v[:,1])
    c=((l[:,3]+l[:,2])*(nu[:,3]-nu[:,2])-(l[:,3]-l[:,2])*(nu[:,3]+nu[:,2]))/(m*np.sqrt(m*m+qt*qt))
    c=np.clip(np.where(v[:,2]<0,-c,c),-1,1)
    ev={k:z[k][fid].astype(float) for k in ['m_w','y_w','pt_w','pt_l','eta_l','mt','weight']}
    ev['y_abs']=np.abs(ev['y_w']);ev['ctw']=c
    p=ev['weight']/ev['weight'].sum()
    meta=dict(path=str(fn),sha256=sha(fn),n_total=int(z['n_total']),n_fid=len(p),
              prior_sigma_fid_pb=float(ev['weight'].sum()/z['n_total']),
              flags=z['flags'].tolist(),fiducial_index_sha256=hashlib.sha256(np.where(fid)[0].tobytes()).hexdigest())
    meta['generation']=dict(hard_process='MadGraph SM LO pp > mu+ vm or mu- vm~',
        shower='Pythia 8.312, power shower, matrix-element corrections off',pdf='CT18NNLO, LHAPDF ID 14000',
        MW_GeV=80.419,GammaW_GeV=2.0476,GF_GeVminus2=1.16639e-5,
        QED_shower=False,primordial_kT_hard_GeV=1.,
        stored_in_fid_note='Stored flag uses pT_l>25 GeV. Study recomputes pT_l>20 GeV and every remaining cut.',
        electroweak_note='Prior parameters differ from fixed order; its mass moments are included among the production constraints.')
    # No validation distributions are formed here. These arrays are kept only
    # to establish fixed event ordering for subsequent prediction.
    return ev,p,meta

def load_training(charge):
    fn=ROOT/'input'/('T_%s.npz'%charge)
    z=np.load(str(fn),allow_pickle=False)
    meta=json.loads(str(z['metadata']));labels=meta['labels'];idx={x:i for i,x in enumerate(labels)}
    m=z['mean'];cov=z['cov'];n=m[0,0]
    r=m/m[0]
    j=np.eye(len(m))/n;j[:,0]-=m[:,0]/n**2
    cr=j@cov@j.T
    return dict(mean=m,cov=cov,r=r,cr=cr,idx=idx,meta=meta,sha=sha(fn))

def tower_resolution(ev,p,t):
    r=t['r'];cr=t['cr'];idx=t['idx'];out={};orders={}
    for name,prefix,cs,den,floor in [
        ('m_w','prof_mw_',cheb(umap(ev['m_w'],40,1000,('bw',80.385,2.085)),12),None,0),
        ('y_abs','prof_absyw_',cheb(umap(ev['y_abs'],0,4,'lin'),12),None,0),
        ('pt_w','prof_wptw_',cheb(umap(ev['pt_w'],25,500,'log'),20),'prof_wptw_0',1)]:
        indices=[idx[prefix+str(n)] for n in range(1,cs.shape[1])]
        if den is None:
            targets=r[indices];stat=np.sqrt(np.maximum(np.diag(cr)[indices],0));prior=p@cs[:,1:]
        else:
            d=idx[den];targets=r[indices]/r[d]
            j=np.zeros((len(indices),len(r)))
            j[np.arange(len(indices)),indices]=1/r[d,0];j[:,d]-=r[indices,0]/r[d,0]**2
            stat=np.sqrt(np.maximum(np.diag(j@cr@j.T),0))
            w=profile_w(ev['pt_w'],25,50,500)
            prior=(p*w)@cs[:,1:]/(p@w)
        scale=.5*np.ptp(targets,axis=1)
        snr=np.abs(targets[:,0]-prior)/np.maximum(np.hypot(stat,scale),1e-30)
        hit=np.where(snr>1)[0]+1;order=max(floor,int(hit.max()) if len(hit) else 0)
        orders[name]=order
        out[name]=dict(order=order,snr=snr.tolist(),stat=stat.tolist(),scale_half_range=scale.tolist(),
                       target=targets.tolist(),prior=prior.tolist())
    return orders,out

def make_features(ev,p,t,orders,model):
    """Return Phi, affine target b+L r, and prior dependence of blended targets.

    prior_target_columns store derivatives d(target)/d p_i evaluated at the
    central fixed-order vector. They are needed for finite-prior statistics.
    """
    idx=t['idx'];d=len(idx);cols=[];rows=[];offset=[];names=[];pd=[]
    def add(name,x,row=None,b=0.,dp=None):
        cols.append(x);rows.append(np.zeros(d) if row is None else row);offset.append(b);names.append(name)
        pd.append(np.zeros(len(p)) if dp is None else dp)
    def unit(tag):
        row=np.zeros(d);row[idx[tag]]=1;return row
    cm=cheb(umap(ev['m_w'],40,1000,('bw',80.385,2.085)),orders['m_w'])
    cy=cheb(umap(ev['y_abs'],0,4,'lin'),max(orders['y_abs'],4))
    for n in range(1,orders['m_w']+1):add('m_T%d'%n,cm[:,n],unit('prof_mw_%d'%n))
    for n in range(1,orders['y_abs']+1):add('y_T%d'%n,cy[:,n],unit('prof_absyw_%d'%n))
    w=profile_w(ev['pt_w'],25,50,500);den=1-p@w;inside=(ev['pt_w']<500).astype(float)
    cp=cheb(umap(ev['pt_w'],25,500,'log'),orders['pt_w'])
    rate=t['r'][idx['prof_wptw_0'],0]
    for n in range(1,orders['pt_w']+1):
        x=cp[:,n]*inside;g=(1-w)*x;part=p@g/den
        row=unit('prof_wptw_%d'%n)-part*unit('prof_wptw_0')
        dp=(1-rate)/den*(g-part*(1-w))
        add('pt_T%d'%n,x,row,part,dp)
    if np.any(inside<1):
        above=1-inside;part=p@above/den
        add('pt_above',above,-part*unit('prof_wptw_0'),part,(1-rate)/den*(above-part*(1-w)))
    cw=profile_w(ev['ctw']+1,.3,.4,1.6,1.7);cc=cheb(ev['ctw'],6)
    if model!='B':
        for n in range(7):add('c_T%d'%n,cw*cc[:,n],unit('prof_wctw_%d'%n))
    degree={'B':0,'A':0,'J2':2,'J4':4}[model]
    for k in range(1,degree+1):
        for n in range(degree+1):
            add('yc_%d%d'%(k,n),cw*cy[:,k]*cc[:,n],unit('prof_yc_%d%d'%(k,n)))
    diag=np.column_stack([cw*cy[:,k]*cc[:,n] for k in range(1,5) for n in range(5)])
    selector=np.stack([unit('prof_yc_%d%d'%(k,n)) for k in range(1,5) for n in range(5)])
    return np.column_stack(cols),np.asarray(offset),np.stack(rows),names,np.column_stack(pd),diag,selector

def linear_residual_cov(phi,p,q,pdep,diag,L,selector,cr,lam):
    f=q@phi;h=q@diag;fc=phi-f;hc=diag-h
    hess=(q[:,None]*fc).T@fc
    sd2=p@(phi-p@phi)**2
    hess+=L2*np.diag(sd2)
    # The solver standardizes using the empirical prior. Its ridge therefore
    # also changes under prior resampling; include that derivative explicitly.
    physical_lam=lam/np.maximum(np.sqrt(sd2),1e-12)
    pdep=pdep-L2*((phi-p@phi)**2-sd2)*physical_lam
    cross=(q[:,None]*hc).T@fc
    response=np.linalg.solve(hess,cross.T).T
    derivative=response@L-selector
    cfo=derivative@cr@derivative.T
    influence=q[:,None]*(hc-fc@response.T)+p[:,None]*(pdep@response.T)
    cprior=influence.T@influence
    return cfo+cprior,cfo,cprior,response,hess

def block_test(residuals,cov,keep):
    if not len(keep):return dict(rank=0,chi2=[0.]*7,p_values=[1.]*7,resolved=False)
    c=cov[np.ix_(keep,keep)];sd=np.sqrt(np.maximum(np.diag(c),1e-30));corr=c/np.outer(sd,sd)
    vals,vec=np.linalg.eigh(corr);retain=vals>1.e-6*max(vals.max(),1e-30)
    rank=int(retain.sum());r=residuals[:,keep]/sd
    chi=((r@vec[:,retain])**2/vals[retain]).sum(axis=1)
    ps=chi2_sf(chi,rank) if rank else np.ones(7)
    return dict(rank=rank,dimension=len(keep),eigenvalues=vals.tolist(),chi2=chi.tolist(),
                p_values=ps.tolist(),resolved=bool(rank and np.max(ps)<.01),
                criterion='resolved only if all seven coherent scale residuals have p<0.01 using central statistical covariance')

def train(charge):
    started=time.time();ev,p,pmeta=load_prior(charge);t=load_training(charge)
    orders,resolution=tower_resolution(ev,p,t)
    print(charge,'events',len(p),'orders',orders,flush=True)
    out=ROOT/'output'/charge;out.mkdir(parents=True,exist_ok=True)
    summary=dict(charge=charge,prior=pmeta,input_sha256=t['sha'],orders=orders,resolution=resolution,
                 l2=L2,models={},independent_validation_opened=False)
    choices={}
    for model in MODELS:
        begin=time.time()
        phi,b,L,names,pdep,H,S=make_features(ev,p,t,orders,model)
        weights=[];lams=[];residuals=[];rec=[]
        lam0=None
        for scale in range(7):
            target=b+L@t['r'][:,scale]
            q,lam,ok=_newton_maxent(phi,p,target,l2=L2,lam0=lam0)
            if not ok:raise RuntimeError('%s %s scale %d failed to converge'%(charge,model,scale))
            if scale==0:lam0=lam.copy()
            weights.append(q);lams.append(lam);residuals.append(q@H-S@t['r'][:,scale])
            achieved=q@phi;stat=np.sqrt(np.maximum(np.diag(L@t['cr']@L.T),1e-30))
            rec.append(dict(scale=scale,max_pull=float(np.max(np.abs(achieved-target)/stat)),
                            eff_retained=float((p@p)/(q@q)),max_ratio=float(np.max(q/p)),
                            min_ratio=float(np.min(q/p)),max_abs_residual=float(np.max(np.abs(achieved-target)))))
            print(charge,model,'scale',scale,rec[-1],flush=True)
        residuals=np.asarray(residuals)
        cov,cfo,cp,response,hess=linear_residual_cov(phi,p,weights[0],pdep,H,L,S,t['cr'],lams[0])
        degree={'B':0,'A':0,'J2':2,'J4':4}[model]
        keep=[j for j,(k,n) in enumerate(( (k,n) for k in range(1,5) for n in range(5)))
              if not (degree and k<=degree and n<=degree)]
        test=block_test(residuals,cov,keep)
        choices[model]=test
        fn=out/(model+'_frozen.npz')
        np.savez_compressed(str(fn),weights=np.asarray(weights),lam=np.asarray(lams),p=p,
                            orders=np.array(json.dumps(orders)),feature_names=np.array(names),
                            b=b,L=L,training_r=t['r'],training_cov=t['cr'],residuals=residuals,
                            residual_cov=cov,residual_cov_fo=cfo,residual_cov_prior=cp,
                            diagnostic_targets=S@t['r'],hessian=hess)
        summary['models'][model]=dict(scales=rec,training_test=test,weights_sha256=sha(fn),
                                     file=str(fn),elapsed_s=time.time()-begin,
                                     hessian_condition=float(np.linalg.cond(hess)),
                                     closes_within_one_MC_error=bool(max(x['max_pull'] for x in rec)<=1.))
        (out/'training_progress.json').write_text(json.dumps(summary,indent=2)+'\n')
        del phi,pdep,H,weights
    selected='A' if not choices['A']['resolved'] else ('J2' if not choices['J2']['resolved'] else 'J4')
    summary['selected']=selected;summary['elapsed_s']=time.time()-started
    summary['selection_feasible']=summary['models'][selected]['closes_within_one_MC_error']
    summary['selection_status']=('frozen candidate closes all supplied targets within one MC error'
        if summary['selection_feasible'] else 'frozen candidate has unresolved target residuals: report this as a fitting limitation')
    summary['freeze_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    (out/'MODEL_FREEZE.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('FROZEN',charge,selected,summary['freeze_utc'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--charge',choices=['WP','WM','both'],default='both');a=ap.parse_args()
    for ch in (['WP','WM'] if a.charge=='both' else [a.charge]):train(ch)
