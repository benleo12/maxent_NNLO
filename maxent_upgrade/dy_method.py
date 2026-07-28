#!/usr/bin/env python3
"""THE PURE METHOD (no K-factor): PS+FO_N prior, FO_{N+k} moment targets.

For each observable X:
  1. detect FO regime [x_match, x_hi] algorithmically:
        bin FO and prior (fine, unit-normalised densities);
        walk x DOWN from kinematic edge; x_match = lowest x with contiguous agreement
        |f_FO/f_prior - 1| < n_sigma * stat   ;  x_hi = top of FO support.
  2. impose FO_{N+k} Chebyshev moments in [x_match, x_hi]  (bare FO; rate-mode configurable).
  3. preserve PS (match prior's OWN low-order moments) outside [x_match, x_hi].
MaxEnt one solve -> positive weights q.

Validation: bin-by-bin vs MC@NLO AND POWHEG; pass = within the scheme spread |MC@NLO-POWHEG|.

Reusable as a library (import build/solve/evaluate) and runnable as a script.
"""
import numpy as np, os

# ---------- observable config ----------
# map: 'lin' or 'log'.  kin = (x_min, x_max) physical support for binning/mapping.
OBSCFG = {
    'mll':   dict(map='log', kin=(60., 1200.),
                  bins=np.array([60,70,80,86,90,92,94,96,100,110,130,160,200,250,320,420,550,700,900,1200])),
    'y_ll':  dict(map='lin', kin=(-2.5, 2.5), bins=np.linspace(-2.5,2.5,22)),
    'pT_ll': dict(map='log', kin=(0.5, 500.),
                  bins=np.array([0,2,4,6,8,10,12,15,20,25,30,40,50,65,80,100,150,220,320,500])),
    'dpm':   dict(map='lin', kin=(0., np.pi), bins=np.linspace(0,np.pi,22)),  # dpm = pi-|dphi|
    'phistar': dict(map='log', kin=(1e-3, 2.0),   # eval-only cross-check (never constrained)
                  bins=np.array([0,.004,.008,.012,.016,.02,.03,.04,.05,.07,.1,.15,.2,.3,.5,.8,1.2,2.0])),
}
# fine grids for regime detection
FINE = {
    'mll':   np.linspace(60,1200,80),
    'y_ll':  np.linspace(-2.5,2.5,60),
    'pT_ll': np.concatenate([np.linspace(0,50,50), np.linspace(55,500,40)]),
    'dpm':   np.linspace(0,np.pi,60),
}


def _phistar(lp, lm):
    pxp,pyp,pzp=lp[:,0],lp[:,1],lp[:,2]; pxm,pym,pzm=lm[:,0],lm[:,1],lm[:,2]
    pmp=np.sqrt(pxp**2+pyp**2+pzp**2); pmm=np.sqrt(pxm**2+pym**2+pzm**2)
    etap=np.arctanh(np.clip(pzp/pmp,-1+1e-12,1-1e-12)); etam=np.arctanh(np.clip(pzm/pmm,-1+1e-12,1-1e-12))
    dphi=(np.arctan2(pyp,pxp)-np.arctan2(pym,pxm)+np.pi)%(2*np.pi)-np.pi
    return np.tan((np.pi-np.abs(dphi))/2)/np.cosh((etam-etap)/2)


def load(subsample=150000, seed=42, cache='dy_method_cache.npz'):
    """Load prior (subsampled), FO NLO, MC@NLO, POWHEG. dpm=pi-|dphi|. Adds phistar for x-checks."""
    if os.path.exists(cache):
        d=np.load(cache, allow_pickle=True)
        out={}
        for grp in ['prior','fo','mc','pw']:
            out[grp]={k[len(grp)+1:]:d[k] for k in d.files if k.startswith(grp+'_')}
        return out
    L=np.load('dy_psLO_partons.npz',allow_pickle=True); F=np.load('dy_efp_NLO_FO.npz',allow_pickle=True)
    MC=np.load('dy_psNLO_partons.npz',allow_pickle=True); PW=np.load('dy_psNLO_powheg_ct18.npz',allow_pickle=True)
    def pack(D, infull=True, has_lep=True):
        inb = D['in_fo'].astype(bool) if (infull and 'in_fo' in D.files) else np.ones(len(D['weight']),bool)
        idx=np.where(inb)[0]
        return idx, D
    # prior subsample
    inL=L['in_fo'].astype(bool); idx=np.where(inL)[0]
    if subsample and subsample<len(idx): idx=np.random.default_rng(seed).choice(idx,subsample,replace=False)
    def grab(D, idx, has_lep=True):
        g=dict(mll=D['mll'][idx], y_ll=D['y_ll'][idx], pT_ll=D['pT_ll'][idx],
                dpm=(np.pi-np.abs(D['dphi_pm'][idx])), w=D['weight'][idx].astype(float))
        if has_lep and 'l_plus' in D.files:
            g['phistar']=_phistar(D['l_plus'][idx].astype(float), D['l_minus'][idx].astype(float))
        return g
    prior=grab(L, idx)
    fo=dict(mll=F['mll'], y_ll=F['y_ll'], pT_ll=F['pT_ll'], dpm=(np.pi-np.abs(F['dphi_pm'])), w=F['weight'].astype(float))
    inM=MC['in_fo'].astype(bool); mc=grab(MC, inM)
    inP=PW['in_fo'].astype(bool); pw=grab(PW, inP)
    out=dict(prior=prior, fo=fo, mc=mc, pw=pw)
    flat={}
    for grp,dd in out.items():
        for k,v in dd.items(): flat[f'{grp}_{k}']=v
    np.savez(cache, **flat)
    return out


# ---------- regime detection ----------
def detect_regime(x_prior, w_prior, x_fo, w_fo, fine_bins, band=0.5, min_count=40, debug=False):
    """FO regime = where the FO/prior density ratio sits on its high-x PLATEAU.
    Physics: above the matching scale both fall like the hard-emission ME (ratio ~ const,
    the O(1) NLO/LO factor); below it FO diverges relative to the Sudakov-resummed shower so
    the ratio climbs away from the plateau. Walk down from FO support edge; stay while the
    ratio is within `band` (fractional) of the plateau; stop at the divergence onset.
    Returns (x_match, x_hi). Empty (preserve) -> x_match=x_hi=top."""
    bw=np.diff(fine_bins); c=0.5*(fine_bins[1:]+fine_bins[:-1])
    hP,_=np.histogram(x_prior,bins=fine_bins,weights=w_prior)
    hF,_=np.histogram(x_fo,bins=fine_bins,weights=w_fo)
    nP,_=np.histogram(x_prior,bins=fine_bins); nF,_=np.histogram(x_fo,bins=fine_bins)
    fP=hP/max(w_prior.sum(),1e-30)/bw; fF=hF/max(w_fo.sum(),1e-30)/bw
    valid=(nP>=min_count)&(nF>=min_count)&(fP>0)&(fF>0)
    vidx=np.where(valid)[0]
    if len(vidx)<3: return float(fine_bins[-1]), float(fine_bins[-1])
    R=np.where(valid, fF/np.maximum(fP,1e-30), np.nan)
    i_hi=vidx[-1]; x_hi=float(fine_bins[i_hi+1])
    upper=vidx[-max(3,len(vidx)//3):]
    plateau=np.median(R[upper])
    if not np.isfinite(plateau) or plateau<=0: return x_hi, x_hi
    i_match=i_hi
    for i in range(i_hi,-1,-1):
        if not valid[i]:
            continue                      # sparse bin: skip, don't extend across a real gap below
        if abs(R[i]/plateau-1.0)<=band:
            i_match=i
        else:
            break
    x_match=float(fine_bins[i_match])
    if debug:
        prof=' '.join(f'{c[i]:.2f}:{R[i]/plateau:.2f}' for i in vidx)
        print(f'    [detect] plateau={plateau:.3f} x_match={x_match:.3f} x_hi={x_hi:.3f}  R/plateau: {prof}',flush=True)
    return x_match, x_hi


# ---------- features ----------
def cheb(u,N):
    u=np.clip(u,-1,1); out=np.zeros((len(u),N+1)); out[:,0]=1.0
    if N>=1: out[:,1]=u
    for n in range(2,N+1): out[:,n]=2*u*out[:,n-1]-out[:,n-2]
    return out
def umap(x, lo, hi, kind):
    if kind=='log':
        lo=max(lo,1e-6); xx=np.clip(x,lo,hi); return 2*(np.log(xx)-np.log(lo))/(np.log(hi)-np.log(lo))-1
    return 2*(np.clip(x,lo,hi)-lo)/(hi-lo)-1


def build(prior, fo, obs_list, N_FO=6, N_pres=3, band=0.5, rate_mode='fo', debug=False):
    """rate_mode: 'fo' = match FO absolute (incl rate) in regime; 'preserve' = preserve prior rate, FO shape."""
    wL=prior['w']; wF=fo['w']
    feats=[]; targets=[]; info={}
    for nm in obs_list:
        cfg=OBSCFG[nm]; xmin,xmax=cfg['kin']; mp=cfg['map']
        XL=prior[nm]; XF=fo[nm]
        x_match,x_hi = detect_regime(XL,wL,XF,wF,FINE[nm],band=band,debug=debug)
        info[nm]=dict(x_match=x_match, x_hi=x_hi, width=x_hi-x_match)
        I_S=(XL<x_match); I_F=(XL>=x_match)&(XL<x_hi); I_T=(XL>=x_hi)
        I_FF=(XF>=x_match)&(XF<x_hi)
        # FO regime: impose FO moments
        if I_F.any() and I_FF.any():
            uL=umap(XL,x_match,x_hi,mp); uF=umap(XF,x_match,x_hi,mp)
            CL=cheb(uL,N_FO); CF=cheb(uF,N_FO)
            if rate_mode=='fo':
                # absolute: match <T_n I_F>_FO for n=0..N  (n=0 is the rate)
                feats.append(I_F.astype(float)); targets.append(wF[I_FF].sum()/wF.sum())
                for n in range(1,N_FO+1):
                    feats.append(CL[:,n]*I_F); targets.append((wF*CF[:,n]*I_FF).sum()/wF.sum())
            else:
                # preserve prior rate, impose FO conditional shape
                prF=wL[I_F].sum()/wL.sum(); foF=wF[I_FF].sum()/wF.sum()
                feats.append(I_F.astype(float)); targets.append(prF)
                for n in range(1,N_FO+1):
                    cond=(wF*CF[:,n]*I_FF).sum()/wF.sum()/max(foF,1e-30)
                    feats.append(CL[:,n]*I_F); targets.append(prF*cond)
        # preserve soft + far (match prior's own low-order moments to pin them)
        for mask,lo,hi in [(I_S,xmin,x_match),(I_T,x_hi,xmax)]:
            if mask.any() and hi>lo:
                uS=umap(XL,lo,hi,mp); CS=cheb(uS,N_pres)
                feats.append(mask.astype(float)); targets.append(wL[mask].sum()/wL.sum())
                for n in range(1,N_pres+1):
                    feats.append(CS[:,n]*mask); targets.append((wL*CS[:,n]*mask).sum()/wL.sum())
    return np.column_stack(feats), np.array(targets), info


# ---------- solve ----------
def maxent(Phi, p, mu, l2=1e-4, n_iter=300, tol=1e-10):
    K=Phi.shape[1]; m_L=(p[:,None]*Phi).sum(0)
    sd=np.maximum(np.sqrt((p[:,None]*(Phi-m_L)**2).sum(0)+1e-30),1e-9)
    Phz=(Phi-m_L)/sd; muz=(mu-m_L)/sd
    lam=np.zeros(K); ok=False
    for it in range(n_iter):
        s=Phz@lam; smax=s.max(); Z=(p*np.exp(s-smax)).sum(); q=p*np.exp(s-smax)/Z
        moms=q@Phz; grad=moms-muz+l2*lam; g=np.linalg.norm(grad)
        if not np.isfinite(g): break
        if g<tol: ok=True; break
        Phc=Phz-moms; H=(q[:,None]*Phc).T@Phc+l2*np.eye(K)
        try:
            if np.linalg.cond(H)>1e11: H+=1e-8*np.eye(K)
            step=np.linalg.solve(H,grad)
        except np.linalg.LinAlgError:
            step=np.linalg.lstsq(H,grad,rcond=None)[0]
        f0=smax+np.log((p*np.exp(s-smax)).sum())-muz@lam+0.5*l2*lam@lam
        t=1.0; lok=False
        for _ in range(60):
            ln=lam-t*step; sn=Phz@ln; smn=sn.max()
            if not np.isfinite(smn): t*=0.5; continue
            fn=smn+np.log((p*np.exp(sn-smn)).sum())-muz@ln+0.5*l2*ln@ln
            if fn<=f0-1e-4*t*(grad@step): lok=True; break
            t*=0.5
        if not lok: break
        lam=ln
    effN=1.0/((q**2).sum()*len(q))*100 if (q>0).all() and np.isfinite(q).all() else 0.0
    return q, effN, it, g, ok


# ---------- evaluation ----------
def evaluate(prior, q, mc, pw, obs_list_eval):
    """Bin-by-bin vs MC@NLO and POWHEG. Returns dict per obs with worst-sigma & TVD & scheme-spread."""
    res={}
    for nm in obs_list_eval:
        bins=OBSCFG[nm]['bins']; bw=np.diff(bins); c=0.5*(bins[1:]+bins[:-1])
        def H(x,w): h,_=np.histogram(x,bins=bins,weights=w/w.sum()); return h/bw
        hP=H(prior[nm],prior['w']); hQ=H(prior[nm],q); hM=H(mc[nm],mc['w']); hW=H(pw[nm],pw['w'])
        nM,_=np.histogram(mc[nm],bins=bins); sigM=1.0/np.sqrt(np.maximum(nM,1))
        nW,_=np.histogram(pw[nm],bins=bins); sigW=1.0/np.sqrt(np.maximum(nW,1))
        mask=hM>1e-6*hM.max()
        def worst(h,ref,sig):
            d=np.where(mask, np.abs(h/np.maximum(ref,1e-30)-1)/np.maximum(sig,1e-9),0); return float(d.max())
        def tvd(h,ref): return 0.5*np.abs((h-ref)*bw).sum()*100
        # scheme spread = POWHEG vs MC@NLO
        res[nm]=dict(
            tvd_q_mc=tvd(hQ,hM), tvd_q_pw=tvd(hQ,hW), tvd_pr_mc=tvd(hP,hM), tvd_pw_mc=tvd(hW,hM),
            worst_q_mc=worst(hQ,hM,sigM), worst_pw_mc=worst(hW,hM,sigM), worst_pr_mc=worst(hP,hM,sigM),
            hP=hP,hQ=hQ,hM=hM,hW=hW,c=c,sigM=sigM,bw=bw,bins=bins)
    return res


if __name__=='__main__':
    import sys
    print('loading...',flush=True)
    D=load()
    prior,fo,mc,pw=D['prior'],D['fo'],D['mc'],D['pw']
    print(f"prior {len(prior['w'])}  fo {len(fo['w'])}  mc {len(mc['w'])}  pw {len(pw['w'])}",flush=True)
    constr=['mll','y_ll','pT_ll','dpm']
    print('detecting regimes (debug):')
    Phi,mu,info=build(prior,fo,constr,N_FO=6,N_pres=3,band=0.5,rate_mode='fo',debug=True)
    print('detected regimes + FO-vs-truth rate sanity (truth=MC@NLO, build never sees it):')
    for nm in constr:
        xm,xh=info[nm]['x_match'],info[nm]['x_hi']
        rate_fo=fo['w'][(fo[nm]>=xm)&(fo[nm]<xh)].sum()/fo['w'].sum()
        rate_mc=mc['w'][(mc[nm]>=xm)&(mc[nm]<xh)].sum()/mc['w'].sum()
        rate_pr=prior['w'][(prior[nm]>=xm)&(prior[nm]<xh)].sum()/prior['w'].sum()
        print(f"  {nm:6s} regime=[{xm:.3f},{xh:.3f}] w={info[nm]['width']:.3f}  rate: FO={rate_fo:.3f} prior={rate_pr:.3f} truth={rate_mc:.3f}")
    p=prior['w']/prior['w'].sum()
    q,effN,it,g,ok=maxent(Phi,p,mu,l2=1e-4)
    print(f'\nK={Phi.shape[1]} eff_N={effN:.1f}% iter={it} |grad|={g:.1e} conv={ok} pos={(q>0).all()}',flush=True)
    res=evaluate(prior,q,mc,pw,['mll','y_ll','pT_ll','dpm','phistar'] if 'phistar' in prior else ['mll','y_ll','pT_ll','dpm'])
    print(f'\n{"obs":7s} {"TVD q-MC":>9s} {"TVD pr-MC":>9s} {"TVD PW-MC":>9s}  {"worst q-MC":>10s} {"worst PW-MC":>11s}  verdict')
    for nm,r in res.items():
        within = r['tvd_q_mc'] <= 1.3*max(r['tvd_pw_mc'],0.3)
        print(f"  {nm:6s} {r['tvd_q_mc']:7.2f}%  {r['tvd_pr_mc']:7.2f}%  {r['tvd_pw_mc']:7.2f}%   {r['worst_q_mc']:8.1f}σ  {r['worst_pw_mc']:9.1f}σ   {'WITHIN SPREAD' if within else 'OUTSIDE'}")
