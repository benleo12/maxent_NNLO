"""Evaluate previously frozen W weights on independent fixed-order predictions."""
from pathlib import Path
import argparse,json,time
import numpy as np
from train import (ROOT,L2,MODELS,load_prior,load_training,make_features,sha,_newton_maxent,chi2_sf)
from stat_utils import bootstrap_raw,NBOOT

def observations(ev,meta):
    cols=[np.ones(len(ev['pt_l']))]
    for tag in ['eta20','eta27','eta30','ptl','mt','recoil']:
        edges=np.asarray(meta['edges'][tag]);x={'ptl':ev['pt_l'],'mt':ev['mt'],'recoil':ev['pt_w']}.get(tag,np.abs(ev['eta_l']))
        cut=ev['pt_l']>int(tag[3:]) if tag.startswith('eta') else np.ones(len(x),bool)
        for lo,hi in zip(edges[:-1],edges[1:]):cols.append(((x>=lo)&(x<hi)&cut).astype(float))
    h=np.column_stack(cols)
    assert h.shape[1]==len(meta['labels'])
    return h

def spectral_stat(res,cov):
    sd=np.sqrt(np.maximum(np.diag(cov),1e-30));c=cov/np.outer(sd,sd)
    v,u=np.linalg.eigh(c);ok=v>1e-6*max(v.max(),1e-30);rank=int(ok.sum())
    score=float(np.sum(((res/sd)@u[:,ok])**2/v[ok]))
    return dict(chi2=score,rank=rank,p_value=float(chi2_sf(score,rank)) if rank else None)

def ratio(x,cov,numerator,denominator):
    a=x[numerator].sum();b=x[denominator].sum();j=np.zeros(len(x))
    j[numerator]+=1/b;j[denominator]-=a/b**2
    return float(a/b),float(np.sqrt(max(j@cov@j,0)))

def input_derivative(phi,p,q,h,L,t,pdep,lam):
    f=q@phi;mean=q@h;fc=phi-f;hc=h-mean
    sd2=p@(phi-p@phi)**2
    hess=(q[:,None]*fc).T@fc+L2*np.diag(sd2)
    pdep=pdep-L2*((phi-p@phi)**2-sd2)*(lam/np.maximum(np.sqrt(sd2),1e-12))
    response=np.linalg.solve(hess,((q[:,None]*hc).T@fc).T).T
    raw=t['mean'][:,0];sigma=raw[0]
    jr=np.eye(len(raw))/sigma;jr[:,0]-=raw/sigma**2
    derivative=sigma*response@L@jr;derivative[:,0]+=mean
    influence=sigma*(q[:,None]*(hc-fc@response.T)+p[:,None]*(pdep@response.T))
    return derivative,influence.T@influence,response,influence

def validate_charge(charge):
    od=ROOT/'output'/charge;freeze=json.loads((od/'MODEL_FREEZE.json').read_text())
    assert not freeze['independent_validation_opened']
    for m in MODELS:assert sha(od/(m+'_frozen.npz'))==freeze['models'][m]['weights_sha256']
    ev,p,pmeta=load_prior(charge);assert pmeta['sha256']==freeze['prior']['sha256']
    t=load_training(charge);assert t['sha']==freeze['input_sha256']
    z=np.load(str(ROOT/'input'/('V_%s.npz'%charge)),allow_pickle=False)
    meta=json.loads(str(z['metadata']));H=observations(ev,meta);ref=z['mean'];refcov=z['cov']
    out=dict(metadata=meta,reference=ref,reference_cov=refcov,selected_model=np.array(freeze['selected']))
    refboot=bootstrap_raw(z['acc'],'V',charge)
    tz=np.load(str(ROOT/'input'/('T_%s.npz'%charge)),allow_pickle=False)
    trainboot=bootstrap_raw(tz['acc'],'T',charge)
    out['_reference_bootstrap']=refboot
    summary=dict(charge=charge,selected=freeze['selected'],freeze_utc=freeze['freeze_utc'],
                 validation_input_sha256=sha(ROOT/'input'/('V_%s.npz'%charge)),models={},
                 statistical_resampling=dict(replicas=NBOOT,scheme='Independent multinomial seed resampling within each fixed-order channel; common draws across all observables. Prediction input response is linear, reference ratios exact; prior statistics use their independent influence covariance.'))
    sigma=t['mean'][0]
    anchor_path=Path('/Users/benassi/reweighting/ref1104/final_v7/%s_central_nomec.npz'%charge.lower())
    anchor=np.load(str(anchor_path),allow_pickle=False)
    for key in ['m_w','y_abs','pt_w','ctw']:
        assert np.allclose(anchor[key],ev[key],rtol=0,atol=1e-12),key
    aq=anchor['trt']/anchor['trt'].sum()
    out['old_blended_angle_anchor_mean']=sigma[0]*(aq@H)
    summary['old_blended_angle_anchor']=dict(path=str(anchor_path),sha256=sha(anchor_path),
        convention='Previously published blended-angle weights, unchanged. Same prior event ordering verified. Normalized to new training baseline rate; original target-input errors are not recomputed here.',
        acceptance={str(cut):float((aq@H)[slice(*meta['slices']['eta%d'%cut])].sum()) for cut in [27,30]})
    anchors={}
    for model in ['P']+MODELS:
        if model=='P':
            means=np.tile(p@H,(7,1))*sigma[:,None]
            D=np.zeros((H.shape[1],len(t['mean'])));D[:,0]=p@H
            influence=sigma[0]*p[:,None]*(H-p@H);cp=influence.T@influence
        else:
            fz=np.load(str(od/(model+'_frozen.npz')),allow_pickle=False)
            weights=fz['weights'];means=weights@H*sigma[:,None]
            phi,b,L,names,pdep,diag,S=make_features(ev,p,t,freeze['orders'],model)
            D,cp,response,influence=input_derivative(phi,p,weights[0],H,L,t,pdep,fz['lam'][0])
            if model==freeze['selected']:
                # Meaningful local check of the linear statistical response:
                # re-solve at +/- one sigma along the two leading raw-covariance
                # eigenvectors, without changing constraints or event selection.
                vals,vec=np.linalg.eigh(t['cov']);checks=[]
                for mode in [-1,-2]:
                    delta=np.sqrt(max(vals[mode],0))*vec[:,mode]
                    for sign in [-1,1]:
                        raw=t['mean'][:,0]+sign*delta;r=raw/raw[0]
                        q,lam,ok=_newton_maxent(phi,p,b+L@r,l2=L2,lam0=fz['lam'][0])
                        if not ok:raise RuntimeError('statistical response check did not converge')
                        actual=raw[0]*(q@H)-means[0];linear=D@(sign*delta)
                        substantial=np.abs(linear)>.05*np.max(np.abs(linear))
                        relative=np.abs(actual-linear)/np.maximum(np.abs(linear),1e-30)
                        checks.append(dict(mode=mode,sign=sign,max_abs_nonlinear_pb=float(np.max(np.abs(actual-linear))),
                                           max_relative_substantial=float(relative[substantial].max()),
                                           median_relative_substantial=float(np.median(relative[substantial]))))
                summary['linear_response_checks']=checks
            del phi,pdep,diag,weights
        ct=D@t['cov']@D.T;cov=ct+cp
        out[model+'_mean']=means;out[model+'_cov_training']=ct;out[model+'_cov_prior']=cp
        out[model+'_fo_derivative']=D
        for anchor,(ad,ai) in anchors.items():
            out[model+'_cov_with_'+anchor+'_training']=D@t['cov']@ad.T
            out[model+'_cov_with_'+anchor+'_prior']=influence.T@ai
        modelboot=means[0]+(trainboot-t['mean'][:,0])@D.T
        out['_'+model+'_training_bootstrap']=modelboot
        rows={}
        for tag in ['eta20','eta27','eta30','ptl','mt']:
            lo,hi=meta['slices'][tag];ix=np.arange(lo,hi)
            diff=means[0,ix]-ref[ix,0];joint=cov[np.ix_(ix,ix)]+refcov[np.ix_(ix,ix)]
            rows[tag]=dict(relative_rms=float(np.sqrt(np.mean((means[0,ix]/ref[ix,0]-1)**2))),
                           statistical_comparison=spectral_stat(diff,joint))
        acc={}
        for cut in [27,30]:
            lo,hi=meta['slices']['eta%d'%cut];ix=np.arange(lo,hi)
            v,ep=ratio(means[0],cp,ix,[0]);vf,ef_delta=ratio(ref[:,0],refcov,ix,[0])
            eb=float((modelboot[:,ix].sum(1)/modelboot[:,0]).std(ddof=1))
            ef=float((refboot[:,ix].sum(1)/refboot[:,0]).std(ddof=1));e=float(np.hypot(ep,eb))
            scale=[ratio(x,cov,ix,[0])[0] for x in means]
            fscale=[ratio(x,refcov,ix,[0])[0] for x in ref.T]
            acc[str(cut)]=dict(value=v,stat=e,stat_training=eb,stat_prior=ep,reference=vf,reference_stat=ef,
                              bias=v-vf,relative_bias=v/vf-1,stat_pull=(v-vf)/np.hypot(e,ef),
                              coherent_scales=scale,reference_coherent_scales=fscale)
        summary['models'][model]=dict(distributions=rows,acceptance=acc)
        print(charge,model,'acceptance',acc,flush=True)
        if model in ['P','A']:anchors[model]=(D.copy(),influence.copy())
    # Correlated changes between fits use the same training draws and prior
    # influence functions; they are not differences of independent errors.
    for model in ['A','J2','J4']:
        if model=='A':continue
        for cut in [27,30]:
            ix=np.arange(*meta['slices']['eta%d'%cut]);xm=out[model+'_mean'][0];xa=out['A_mean'][0]
            jm=np.zeros(len(xm));ja=np.zeros(len(xa))
            jm[ix]=1/xm[0];jm[0]-=xm[ix].sum()/xm[0]**2
            ja[ix]=1/xa[0];ja[0]-=xa[ix].sum()/xa[0]**2
            vp=jm@out[model+'_cov_prior']@jm+ja@out['A_cov_prior']@ja-2*jm@out[model+'_cov_with_A_prior']@ja
            bm=out['_'+model+'_training_bootstrap'];ba=out['_A_training_bootstrap']
            db=bm[:,ix].sum(1)/bm[:,0]-ba[:,ix].sum(1)/ba[:,0]
            delta=float(xm[ix].sum()/xm[0]-xa[ix].sum()/xa[0])
            summary['models'][model]['acceptance'][str(cut)]['difference_from_A']=dict(
                value=delta,stat=float(np.sqrt(db.var(ddof=1)+max(vp,0))),
                stat_training=float(db.std(ddof=1)),stat_prior=float(np.sqrt(max(vp,0))))
    np.savez_compressed(str(od/'predictions.npz'),**{k:(np.array(json.dumps(v)) if isinstance(v,dict) else v) for k,v in out.items() if not k.startswith('_')})
    (od/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    return out,summary

def charge_asymmetry(results):
    p,n=results['WP'],results['WM'];meta=p['metadata'];out={}
    for model in ['reference','P']+MODELS+['selected']:
        if model=='reference':
            mp=p['reference'].T;mn=n['reference'].T;cp=p['reference_cov'];cn=n['reference_cov']
            bp=p['_reference_bootstrap'];bn=n['_reference_bootstrap'];cpp=np.zeros_like(cp);cnp=np.zeros_like(cn)
        else:
            pmodel=str(p['selected_model']) if model=='selected' else model
            nmodel=str(n['selected_model']) if model=='selected' else model
            mp=p[pmodel+'_mean'];mn=n[nmodel+'_mean'];cp=p[pmodel+'_cov_training']+p[pmodel+'_cov_prior'];cn=n[nmodel+'_cov_training']+n[nmodel+'_cov_prior']
            bp=p['_'+pmodel+'_training_bootstrap'];bn=n['_'+nmodel+'_training_bootstrap']
            cpp=p[pmodel+'_cov_prior'];cnp=n[nmodel+'_cov_prior']
        tags={}
        for tag in ['eta20','eta27','eta30']:
            lo,hi=meta['slices'][tag];ix=np.arange(lo,hi);a,b=mp[:,ix],mn[:,ix]
            asym=(a-b)/(a+b);jp=2*b[0]/(a[0]+b[0])**2;jn=-2*a[0]/(a[0]+b[0])**2
            cov=cp[np.ix_(ix,ix)]*np.outer(jp,jp)+cn[np.ix_(ix,ix)]*np.outer(jn,jn)
            ab=(bp[:,ix]-bn[:,ix])/(bp[:,ix]+bn[:,ix]);cpstat=cpp[np.ix_(ix,ix)]*np.outer(jp,jp)+cnp[np.ix_(ix,ix)]*np.outer(jn,jn)
            stat=np.sqrt(ab.var(axis=0,ddof=1)+np.maximum(np.diag(cpstat),0))
            tags[tag]=dict(coherent_scales=asym.tolist(),cov=cov.tolist(),stat=stat.tolist())
        out[model]=tags
    for model in ['P']+MODELS+['selected']:
        for tag in ['eta20','eta27','eta30']:
            r=out[model][tag];f=out['reference'][tag]
            diff=np.array(r['coherent_scales'])[0]-np.array(f['coherent_scales'])[0]
            r['rms_absolute_bias']=float(np.sqrt(np.mean(diff**2)))
            r['statistical_comparison']=spectral_stat(diff,np.array(r['cov'])+np.array(f['cov']))
    def model_difference(model,anchor,tag):
        pm=str(p['selected_model']) if model=='selected' else model
        nm=str(n['selected_model']) if model=='selected' else model
        lo,hi=meta['slices'][tag];ix=np.arange(lo,hi)
        mp=p[pm+'_mean'][:,ix];mn=n[nm+'_mean'][:,ix]
        ap=p[anchor+'_mean'][:,ix];an=n[anchor+'_mean'][:,ix]
        scales=(mp-mn)/(mp+mn)-(ap-an)/(ap+an)
        cov=np.zeros((len(ix),len(ix)));prior_cov=cov.copy()
        for cache,name,jm,ja in [
            (p,pm,2*mn[0]/(mp[0]+mn[0])**2,2*an[0]/(ap[0]+an[0])**2),
            (n,nm,-2*mp[0]/(mp[0]+mn[0])**2,-2*ap[0]/(ap[0]+an[0])**2)]:
            for source in ['training','prior']:
                cm=cache[name+'_cov_'+source][np.ix_(ix,ix)]
                ca=cache[anchor+'_cov_'+source][np.ix_(ix,ix)]
                cross=cm if name==anchor else cache[name+'_cov_with_'+anchor+'_'+source][np.ix_(ix,ix)]
                cd=cm*np.outer(jm,jm)+ca*np.outer(ja,ja)-cross*np.outer(jm,ja)-cross.T*np.outer(ja,jm)
                cov+=cd
                if source=='prior':prior_cov+=cd
        bp=p['_'+pm+'_training_bootstrap'][:,ix];bn=n['_'+nm+'_training_bootstrap'][:,ix]
        bap=p['_'+anchor+'_training_bootstrap'][:,ix];ban=n['_'+anchor+'_training_bootstrap'][:,ix]
        delta=(bp-bn)/(bp+bn)-(bap-ban)/(bap+ban)
        stat=np.sqrt(delta.var(axis=0,ddof=1)+np.maximum(np.diag(prior_cov),0))
        return dict(coherent_scales=scales.tolist(),cov=cov.tolist(),stat=stat.tolist(),
                    definition='Paired model difference: common training resamples and cross-model finite-prior influence covariance, including fit response.')
    for model in ['P']+MODELS+['selected']:
        for tag in ['eta20','eta27','eta30']:
            out[model][tag]['difference_from_prior']=model_difference(model,'P',tag)
    for model in ['J2','J4','selected']:
        for tag in ['eta20','eta27','eta30']:
            out[model][tag]['difference_from_angle']=model_difference(model,'A',tag)
    return out

if __name__=='__main__':
    # Both charges must be frozen before either validation prediction is opened.
    for ch in ['WP','WM']:assert (ROOT/'output'/ch/'MODEL_FREEZE.json').exists()
    results={};summaries={}
    for ch in ['WP','WM']:results[ch],summaries[ch]=validate_charge(ch)
    asym=charge_asymmetry(results)
    (ROOT/'output'/'charge_asymmetry.json').write_text(json.dumps(asym,indent=2)+'\n')
    (ROOT/'output'/'VALIDATION_COMPLETE.json').write_text(json.dumps(dict(
        completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        selected={ch:summaries[ch]['selected'] for ch in summaries},
        caveat='Input-MC and finite-prior errors are linear-response estimates; scale envelopes remain separate. No uncertainty coverage claim.'),indent=2)+'\n')
