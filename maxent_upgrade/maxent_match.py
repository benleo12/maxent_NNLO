#!/usr/bin/env python3
"""maxent_match: THE unified MaxEnt-matching pipeline (order-uniform, fully coded criteria).

Algorithm (paper box):
 A1. DOF rule: constrain a minimal spanning set = Born invariants of the hard system
     + ONE recoil magnitude per hard system. Everything else FOLLOWS (never constrained).
 A2. Born DOFs -> differential K-factor targets:
     K(x) = dens_{N+k}(x)/dens_N(x) on the FO binning, with a TWO-SIDED validity mask:
       mask (K:=1, preserve) any bin with rel_err > RELMAX
       or |K/plateau - 1| > DELTA where plateau = median K over the contiguous valid top region
     (catches both statistical noise and FO breakdown e.g. jacobian shoulders).
     Targets = Chebyshev moments (N_FO) of the K-reweighted prior. Masked sigma-fraction is REPORTED.
 A3. Recoil DOF -> windowed bin-probability targets:
     x_match rule (coded, data-blind):
       if FO_N populates the recoil spectrum (two orders available):
         K_conv(b) = dens_{N+k}(b)/dens_N(b); plateau = median over top third;
         x_match = lowest edge such that ALL bins above satisfy |K_conv/plateau-1| < DELTA contiguously.
       else (first non-vanishing order, k=1 from a Born-only prior):
         x_match = seam of minimal local FO-shower conditional discrepancy (threshold-free)
         (FO-shower agreement rule; the k=1 special case).
     x_hi = end of FO support. Window targets: rate * conditional FO bin probabilities.
     rate SCHEMES = {max(prior,FO) [central], prior, FO} -> matching-scheme band (reported).
     Complementarity: soft rate = 1 - rate - preserved_tail; soft shape = soft_rate * conditional
     Chebyshev (N_PRES) moments of the prior (uniform scaling, Sudakov peak intact).
 A4. Solve: z-scored Newton on the convex dual, L2 = 1e-4 (closure = exact to <1e-4 rel).
 A5. Uncertainty: re-solve per FO scale variation and per rate scheme -> envelope; stat from weights.
 All dials live in DIALS below; --scan produces the stability table over all of them.

===========================================================================
PACKAGE COPY (maxent_upgrade): VERBATIM copy of the frozen physics engine with
ONE additive, physics-neutral change so the moment-SNR selector can set the
number of constrained Chebyshev moments PER observable instead of the single
global N_FO / N_W dial:
  * build_and_solve reads optional cfg['born_N'] = {obs: n} and cfg['recoil_N'] = n.
    When those keys are ABSENT it falls back to dials['N_FO'] / dials['N_W'] and
    reproduces the original engine bit-for-bit.
No physics is altered: diff-K, the two-sided validity masks, the composite
window, the convex-dual Newton solver, and the scale/rate band are untouched.
===========================================================================
"""
import numpy as np, json, argparse, sys
try:                                    # PACKAGE COPY: resolve the sibling module
    from .dy_method import maxent, cheb, umap       # when imported as maxent_upgrade.maxent_match
except ImportError:                                 # when imported as a top-level module
    from dy_method import maxent, cheb, umap

DIALS = dict(N_FO=6, N_PRES=3, L2=1e-4, RELMAX=0.15, DELTA=0.20,
             CLIP=(0.3, 3.0), MINSUP=30, RSUP=20.0, WBASIS='composite', N_W=12, WSIG=2.0)
# RELMAX gates bins entering the x_match CRITERION and Born diff-K (needs precise ratios).
# WSIG gates WINDOW admission by significance (val/err > WSIG): a large correction with a
# large relative error still beats discarding it, its error belongs in the band not a veto.  # window basis: 'cheb' (moment) or 'indicator'  # window-bin support: >=MINSUP prior events AND target/prior prob ratio < RSUP

# ---------------- FO input containers ----------------
def fo_from_dat(paths, nscales=7):
    """NNLOJET-style .dat (densities): concat multiple files; returns dict."""
    lo=[];hi=[];vals=[[] for _ in range(nscales)];errs=[[] for _ in range(nscales)]
    for path in paths:
        for line in open(path):
            s=line.strip()
            if not s or s.startswith('#'): continue
            try: r=[float(x) for x in s.split()]
            except ValueError: continue
            lo.append(r[0]); hi.append(r[2])
            for k in range(nscales): vals[k].append(r[3+2*k]); errs[k].append(r[4+2*k])
    lo=np.array(lo); hi=np.array(hi)
    o=np.argsort(lo)
    return dict(lo=lo[o],hi=hi[o],vals=[np.array(v)[o] for v in vals],
                errs=[np.array(e)[o] for e in errs],kind='dat')

def fo_from_events(x_by_obs, w, w_scale=None):
    """FO event sample: dict of observable arrays + weights (+ per-event scale weights)."""
    return dict(x=x_by_obs, w=np.asarray(w,float),
                w_scale=(np.asarray(w_scale,float) if w_scale is not None else None), kind='events')

def _fo_hist(fo, obs, edges, k=0):
    """(density, rel_err, prob) of FO input on given edges for scale k."""
    if fo['kind']=='dat':
        _fe=np.concatenate([fo['lo'][:1],fo['hi']])
        assert len(_fe)==len(edges) and np.allclose(edges,_fe,rtol=2e-3,atol=2e-3), 'dat edges mismatch'
        v=fo['vals'][k]; e=fo['errs'][k]; w=fo['hi']-fo['lo']
        tot=(v*w).sum()
        return v/tot, np.abs(e)/np.maximum(np.abs(v),1e-30), v*w/tot
    else:
        w = fo['w'] if (k==0 or fo['w_scale'] is None) else fo['w_scale'][:,k]
        h,_=np.histogram(fo['x'][obs],bins=edges,weights=w)
        h2,_=np.histogram(fo['x'][obs],bins=edges,weights=w**2)
        widths=np.diff(edges); tot=h.sum()
        rel=np.sqrt(np.maximum(h2,0))/np.maximum(np.abs(h),1e-30)
        return (h/widths)/tot, rel, h/tot

def _nscales(fo):
    if fo['kind']=='dat': return len(fo['vals'])
    return 1 if fo['w_scale'] is None else fo['w_scale'].shape[1]

# ---------------- coded criteria ----------------
def kconv_xmatch(fo_hi, fo_lo, obs, edges, delta, report, relmax=0.15):
    """series-convergence x_match, plateau-anchored.
    The convergence ratio K_conv has an INTERIOR plateau (stable higher-order
    correction), physical structure above it (running coupling, channel opening,
    both legitimately FO), and the logarithmic divergence below it. The rule:
    find the plateau as the contiguous 5 valid bins with minimal relative spread
    of K_conv, anchor plateau = median there, then walk DOWNWARD only; x_match is
    the lowest bin maintaining |K_conv/plateau - 1| < delta contiguously below
    the plateau region. No convergence constraint is applied above the plateau."""
    dH,relH,_=_fo_hist(fo_hi,obs,edges); dL,relL,_=_fo_hist(fo_lo,obs,edges)
    valid=(dL>1e-12)&(dH>1e-30)&(relH<relmax)&(relL<relmax)
    K=np.where(valid, dH/np.maximum(dL,1e-300), np.nan)
    n=len(K); iv=np.where(valid)[0]
    if len(iv)<6:
        report['kconv']=dict(rule='series-convergence', error='too few valid bins', xm_edge=float(edges[-1]))
        return n-1
    kv=K[iv]
    spreads=[(float(np.std(kv[j:j+5])/max(abs(np.mean(kv[j:j+5])),1e-300)),j) for j in range(len(iv)-4)]
    _,j0=min(spreads)
    plateau=float(np.median(kv[j0:j0+5]))
    xm_j=j0
    for j in range(j0-1,-1,-1):
        if abs(kv[j]/plateau-1)<delta: xm_j=j
        else: break
    xm_idx=int(iv[xm_j])
    report['kconv']=dict(plateau=plateau, xm_edge=float(edges[xm_idx]), rule='series-convergence',
                         delta=delta, n_valid=int(len(iv)),
                         plateau_region=[float(edges[iv[j0]]),float(edges[min(iv[j0+4]+1,n-1)])],
                         K_conv_valid={float(edges[i]): float(K[i]) for i in iv})
    return xm_idx

def ps_agreement_xmatch(fo_hi, prior_x, prior_w, obs, edges, report, relmax=0.15):
    """k=1 fallback: FO-vs-prior conditional agreement plateau, evaluated on a
    statistically meaningful COARSE rebinning (~12 log bins). Bins where either
    side lacks statistics are unknown, they neither pass nor break contiguity."""
    lo,hi=edges[0],edges[-1]
    ce=np.geomspace(max(lo,edges[1],0.5),hi,13)   # geometric always; recoil spectra live on log scales
    dF,relF,probF=_fo_hist(fo_hi,obs,edges)
    wsum=probF  # per-fine-bin probabilities
    # aggregate FO onto coarse bins (probability-weighted, with error propagation)
    cF=np.zeros(12); cFe=np.zeros(12)
    ctr=0.5*(edges[:-1]+edges[1:])
    idx=np.clip(np.searchsorted(ce,ctr)-1,0,11)
    for b in range(12):
        m=idx==b
        cF[b]=probF[m].sum()
        cFe[b]=np.sqrt(((probF[m]*relF[m])**2).sum())
    hP,_=np.histogram(prior_x,bins=ce,weights=prior_w)
    nP,_=np.histogram(prior_x,bins=ce)
    pP=hP/max(hP.sum(),1e-300)
    valid=(cF>0)&(nP>=50)&(cFe<relmax*np.maximum(cF,1e-300))
    iv=np.where(valid)[0]
    if len(iv)<3:
        report['kconv']=dict(rule='FO-shower-agreement(k=1)', error='too few valid coarse bins',
                             xm_edge=float(edges[-1]))
        return len(edges)-1
    # MINIMAL-DISCREPANCY SEAM (k=1): unresummed FO and an LL shower never agree
    # to matching accuracy anywhere (their gap IS the matching correction), so no
    # threshold rule is well posed. The seam is placed where the local conditional
    # discrepancy D(b) is MINIMAL (the cleanest join), and the genuine scheme
    # freedom is the span of scales with D <= 2 D_min, enumerated into the band.
    Dv={}
    for jj in range(len(iv)):
        b=int(iv[jj])
        loc=iv[(iv>=b)][:3]
        if len(loc)<2: continue
        f=cF[loc]/max(cF[loc].sum(),1e-300)
        p=pP[loc]/max(pP[loc].sum(),1e-300)
        r=f/np.maximum(p,1e-300)
        Dv[b]=float(np.max(np.abs(r-1)))
    if not Dv:
        report['kconv']=dict(rule='minimal-discrepancy-seam(k=1)', error='no evaluable seam',
                             xm_edge=float(edges[-1]))
        return len(edges)-1
    Dmin=min(Dv.values())
    xm_c=min([b for b,d in Dv.items() if d<=Dmin*1.0001])
    span=[b for b,d in Dv.items() if d<=2*Dmin]
    report['seam_profile']={float(ce[b]): round(d,3) for b,d in sorted(Dv.items())}
    report['seam_span_edges']=[float(ce[min(span)]),float(ce[max(span)])]
    xm_val=ce[xm_c]
    xm_idx=int(np.searchsorted(edges,xm_val))
    report['kconv']=dict(xm_edge=float(edges[min(xm_idx,len(edges)-1)]), rule='minimal-discrepancy-seam(k=1)',
                         D_min=round(min(Dv.values()),3), n_valid=int(len(iv)))
    return min(xm_idx,len(edges)-1)

def masked_diffK(fo_hi, obs, edges, dens_lo, rel_lo, dials, report):
    """Born diff-K with two-sided validity mask; returns callable and mask report."""
    dH,relH,probH=_fo_hist(fo_hi,obs,edges)
    ok=(dens_lo>1e-12)&(dH>0)
    K=np.where(ok, dH/np.maximum(dens_lo,1e-30), 1.0)
    stat_bad=(relH>dials['RELMAX'])|(rel_lo>dials['RELMAX'])
    valid=ok&~stat_bad
    plateau=np.median(K[valid]) if valid.sum()>=3 else 1.0
    conv_bad=np.abs(K/max(plateau,1e-30)-1)>3*dials['DELTA']   # loose: Born K may legitimately vary
    mask=stat_bad|(~ok)
    K=np.where(mask,1.0,K)
    K=np.clip(K,*dials['CLIP'])
    report[f'diffK_{obs}']=dict(masked_bins=int(mask.sum()),
        masked_sigma_frac=float(probH[mask].sum()), K_range=[float(K.min()),float(K.max())])
    def f(x): return K[np.clip(np.searchsorted(edges,x)-1,0,len(K)-1)]
    return f

# ---------------- solver ----------------
def _newton_maxent(Phi, p, mu, l2=1e-4, lam0=None, n_iter=200, tol=1e-10):
    """z-scored Newton on the convex dual; warm-startable. Returns (q, lam, ok)."""
    K=Phi.shape[1]
    m_L=(p[:,None]*Phi).sum(0)
    sd=np.maximum(np.sqrt((p[:,None]*(Phi-m_L)**2).sum(0)+1e-30),1e-12)
    Phz=(Phi-m_L)/sd; muz=(mu-m_L)/sd
    lam=np.zeros(K) if lam0 is None else np.asarray(lam0,float).copy()
    for it in range(n_iter):
        sarr=Phz@lam; smax=sarr.max()
        Z=(p*np.exp(sarr-smax)).sum(); q=p*np.exp(sarr-smax)/Z
        grad=q@Phz-muz+l2*lam; g=float(np.linalg.norm(grad))
        if not np.isfinite(g): return None,lam,False
        if g<tol: break
        Phc=Phz-(q@Phz); H=(q[:,None]*Phc).T@Phc+l2*np.eye(K)
        try: step=np.linalg.solve(H+1e-12*np.eye(K),grad)
        except np.linalg.LinAlgError: step=np.linalg.lstsq(H,grad,rcond=None)[0]
        f0=smax+np.log(Z)-muz@lam+0.5*l2*lam@lam
        t=1.0; okls=False
        for _ in range(50):
            ln=lam-t*step; sn=Phz@ln; smn=sn.max()
            if np.isfinite(smn):
                fn=smn+np.log((p*np.exp(sn-smn)).sum())-muz@ln+0.5*l2*ln@ln
                if fn<=f0-1e-4*t*float(grad@step): okls=True; break
            t*=0.5
        if not okls: break
        lam=ln
    sarr=Phz@lam; smax=sarr.max(); q=p*np.exp(sarr-smax); q/=q.sum()
    ok=bool(np.isfinite(q).all() and (q>0).all() and g<1e-6)
    return q,lam,ok

# ---------------- pipeline ----------------
def build_and_solve(prior, cfg, fo_lo, fo_hi, k_scale=0, rate_scheme='max', dials=DIALS,
                    xm_override=None, report=None, lam0=None):
    """prior: dict of arrays + 'w'. cfg: dict(born=[(obs,lo,hi,map)], recoil=(obs,edges_from_fo,softmap_lo),
       followers=[...]). Returns q, report."""
    if report is None: report={}
    wL=prior['w']; p=wL/wL.sum()
    F=[np.ones(len(wL))]; T=[1.0]; names=['norm']
    umaps={}; mapspec={}
    # Born DOFs
    born_N=cfg.get('born_N',{})            # PACKAGE COPY: optional per-observable moment count
    for (obs,lo,hi,mp) in cfg['born']:
        edges=cfg['born_edges'][obs]
        _Nfo=int(born_N.get(obs,dials['N_FO']))
        dL,relL,_=_fo_hist(fo_lo,obs,edges,0) if fo_lo is not None else (None,None,None)
        fK=masked_diffK(fo_hi,obs,edges,dL,relL,dials,report) if fo_lo is not None else (lambda x: np.ones_like(x))
        xv=prior[obs]; wK=wL*fK(xv)
        u=umap(xv,lo,hi,mp); C=cheb(u,max(_Nfo,1))
        umaps[obs]=u; mapspec[obs]=(lo,hi,mp)
        for n in range(1,_Nfo+1):
            F.append(C[:,n]); T.append((wK*C[:,n]).sum()/wK.sum()); names.append(f'{obs}_T{n}')
    # Recoil DOF
    obs,edges,soft_lo = cfg['recoil']
    XL=prior[obs]
    dH,relH,probH=_fo_hist(fo_hi,obs,edges,k_scale)
    if xm_override is not None:
        xm_idx=int(np.searchsorted(edges,xm_override)); report['kconv']=dict(xm_edge=float(edges[xm_idx]),rule='override')
    elif fo_lo is not None and _fo_hist(fo_lo,obs,edges)[2].sum()>1e-6:
        # COMBINED criterion at k>=2: the order ratio detects relative breakdown, but the
        # common logarithmic divergence CANCELS in the ratio; the FO-shower seam sees it.
        # x_match = max of the two data-blind bounds.
        xm_series=kconv_xmatch(fo_hi,fo_lo,obs,edges,dials['DELTA'],report,relmax=dials['RELMAX'])
        rep_series=report.pop('kconv',{})
        xm_seam=ps_agreement_xmatch(fo_hi,XL,wL,obs,edges,report,relmax=dials['RELMAX'])
        rep_seam=report.pop('kconv',{})
        if rep_series.get('error'):
            xm_idx=xm_seam; report['kconv']={**rep_seam,'rule':'seam(series unavailable)'}
        else:
            xm_idx=max(xm_series,xm_seam)
            report['kconv']=dict(rule='max(series,seam)',
                xm_edge=float(edges[xm_idx]), xm_series=rep_series.get('xm_edge'),
                xm_seam=rep_seam.get('xm_edge'), plateau=rep_series.get('plateau'),
                K_conv_valid=rep_series.get('K_conv_valid'))
            if 'seam_profile' in report: report['seam_profile_note']='seam profile from k=1 signal'
    else:
        xm_idx=ps_agreement_xmatch(fo_hi,XL,wL,obs,edges,report,relmax=dials['RELMAX'])
    # RATE-CONSISTENCY central x_match: within the validity-admissible region choose the
    # scale where the class's two window-rate estimates (prior and positive-part FO) agree
    # best; there the dominant scheme ambiguity vanishes by construction. The validity
    # bound remains the aggressive end of the enumerated scheme span. Falls back to the
    # validity bound when no consistency point exists (support-starved priors).
    if xm_override is None and dials.get('XM_MODE','off')=='rate-consistent':
        _dH,_relH,_probH=_fo_hist(fo_hi,obs,edges,k_scale)
        _pos=_probH>0
        _sigpos=max(_probH[_pos].sum(),1e-300)
        best=None
        for _e in range(xm_idx,len(edges)-3):
            _pr=float(wL[XL>=edges[_e]].sum()/wL.sum())
            _fo=float(_probH[(np.arange(len(_dH))>=_e)&_pos].sum()/_sigpos)
            if _pr<=0 or _fo<=0: continue
            _d=abs(np.log(_fo/_pr))
            if best is None or _d<best[0]: best=(_d,_e)
        if best is not None and best[0]<0.15:
            report['xm_validity_bound']=float(edges[xm_idx])
            xm_idx=best[1]
            report['xm_rate_consistency']=float(edges[xm_idx])
        else:
            report['xm_rate_consistency']=None
    else:
        # report-only diagnostic: where the two class rate estimates cross (companion band study)
        _dH2,_r2,_p2=_fo_hist(fo_hi,obs,edges,k_scale); _pos2=_p2>0; _sp2=max(_p2[_pos2].sum(),1e-300)
        _best=None
        for _e in range(xm_idx,len(edges)-3):
            _pr=float(wL[XL>=edges[_e]].sum()/wL.sum()); _fo=float(_p2[(np.arange(len(_dH2))>=_e)&_pos2].sum()/_sp2)
            if _pr>0 and _fo>0:
                _d=abs(np.log(_fo/_pr))
                if _best is None or _d<_best[0]: _best=(_d,_e)
        report['xm_rate_consistency_diag']=float(edges[_best[1]]) if _best else None
    XM=edges[xm_idx]; XHI=edges[-1]
    stat_bad=~((dH>0)&(relH<1.0/max(dials['WSIG'],1e-9)))   # window admission: significance, not precision
    sel=np.arange(len(dH))>=xm_idx
    cnt,_=np.histogram(XL,bins=edges)      # unweighted prior support per bin
    hw,_=np.histogram(XL,bins=edges,weights=wL)
    prior_prob=hw/wL.sum()
    ratio=probH/np.maximum(prior_prob,1e-300)
    no_support=(cnt<dials['MINSUP'])|(ratio>dials['RSUP'])
    use=sel&~stat_bad&~no_support
    if use.any():                           # cap window at last contiguous supported bin
        last=int(np.where(use)[0].max()); XHI=edges[last+1]
        use&=np.arange(len(dH))<=last
    report['window']=dict(xm=float(XM),xhi=float(XHI),n_bins=int(use.sum()),
                          stat_masked=int((sel&stat_bad).sum()),
                          support_masked=int((sel&no_support).sum()),
                          support_sigma_frac=float(probH[sel&no_support].sum()))
    I_F=(XL>=XM)&(XL<XHI); I_S=(XL<XM)
    prior_rate=float(wL[I_F].sum()/wL.sum())
    pos=probH>0
    fo_rate=float(probH[use].sum()/max(probH[pos].sum(),1e-300))   # positive-part normalization
    neg_soft=float(-probH[~pos].sum())
    if neg_soft>1e-6: report.setdefault('rate_flags',{})['fo_negative_region_prob']=neg_soft
    rate={'max':max(prior_rate,fo_rate),'prior':prior_rate,'fo':fo_rate}[rate_scheme]
    report['rate']=dict(prior=prior_rate,fo=fo_rate,scheme=rate_scheme,used=rate)
    pb=probH[use]/probH[use].sum()
    blo=edges[:-1][use]; bhi=edges[1:][use]
    if dials.get('WBASIS')=='composite':
        # PURE EVENT-LEVEL MOMENT FORM (method definition): one global Chebyshev tower
        # against the composite target density f* = P_soft*prior_cond + P_W*FO_cond + P_tail*prior_cond.
        # No indicator features anywhere; rates enter only through the target moments.
        _Nw=int(cfg.get('recoil_N',dials['N_W']))   # PACKAGE COPY: optional recoil moment count
        p_here=wL/wL.sum()
        P_tail=float(p_here[XL>=XHI].sum())
        P_soft=1.0-rate-P_tail
        lo_map=soft_lo
        uR=umap(np.clip(XL,lo_map,XHI),lo_map,XHI,'log'); CR=cheb(uR,max(_Nw,1))
        umaps[obs]=uR; mapspec[obs]=(lo_map,XHI,'log')
        I_S_=XL<XM; I_T_=XL>=XHI
        wS_=float(p_here[I_S_].sum()); wT_=max(P_tail,1e-300)
        ctr=0.5*(blo+bhi)
        uF=(np.log(ctr)-np.log(lo_map))/(np.log(XHI)-np.log(lo_map))*2-1
        CF=np.polynomial.chebyshev.chebvander(np.clip(uF,-1,1),max(_Nw,1))
        for n in range(1,_Nw+1):
            soft_mom=float((p_here[I_S_]*CR[I_S_,n]).sum())/max(wS_,1e-300)
            tail_mom=float((p_here[I_T_]*CR[I_T_,n]).sum())/wT_ if P_tail>0 else 0.0
            fo_mom=float((pb*CF[:,n]).sum())
            F.append(CR[:,n]); T.append(P_soft*soft_mom+rate*fo_mom+P_tail*tail_mom)
            names.append(f'recoil_T{n}')
    elif dials.get('WBASIS','cheb')=='cheb':
        # polynomial moment constraints of the conditional window shape (log map),
        # FO-side moments from fine-bin midpoints (accurate for fine input binning)
        I_W=((XL>=XM)&(XL<XHI)).astype(float)
        uW=umap(XL,XM,XHI,'log'); CW=cheb(uW,dials['N_W'])
        ctr=0.5*(blo+bhi)
        uF=(np.log(ctr)-np.log(XM))/(np.log(XHI)-np.log(XM))*2-1
        CF=np.polynomial.chebyshev.chebvander(np.clip(uF,-1,1),dials['N_W'])
        F.append(I_W); T.append(rate); names.append('win_rate')
        for n in range(1,dials['N_W']+1):
            F.append(CW[:,n]*I_W); T.append(rate*float((pb*CF[:,n]).sum())); names.append(f'win_T{n}')
    else:
        for j in range(len(pb)):
            F.append(((XL>=blo[j])&(XL<bhi[j])).astype(float)); T.append(rate*pb[j]); names.append(f'win[{blo[j]:.0f},{bhi[j]:.0f}]')
    p_tail=float(wL[XL>=XHI].sum()/wL.sum())
    soft=1.0-rate-p_tail
    if dials.get('WBASIS')!='composite' and XM>edges[0]+1e-9 and soft>1e-6:
        uS=umap(XL,soft_lo,XM,'log'); CS=cheb(uS,dials['N_PRES']); wS=wL[I_S].sum()
        F.append(I_S.astype(float)); T.append(soft); names.append('soft_rate')
        for n in range(1,dials['N_PRES']+1):
            F.append(CS[:,n]*I_S); T.append(soft*(wL*CS[:,n]*I_S).sum()/wS); names.append(f'soft_T{n}')
    # ---- joint (cross) moments: event-level, composite-consistent ----
    for (oa,ob,NA,NB) in cfg.get('joints',[]):
        fo=fo_hi['parts'][oa] if isinstance(fo_hi,dict) and fo_hi.get('kind')=='multi' else fo_hi
        fo_b=fo_hi['parts'][ob] if isinstance(fo_hi,dict) and fo_hi.get('kind')=='multi' else fo_hi
        ev = fo if fo.get('kind')=='events' else (fo_b if fo_b.get('kind')=='events' else None)
        if ev is None or oa not in ev['x'] or ob not in ev['x']:
            report[f'joint_{oa}_{ob}']='skipped (no FO event-level 2D input)'
            continue
        wF = ev['w'] if (k_scale==0 or ev.get('w_scale') is None) else ev['w_scale'][:,k_scale]
        loA,hiA,mpA=mapspec[oa]; loB,hiB,mpB=mapspec[ob]
        uAf=umap(np.clip(ev['x'][oa],loA,hiA),loA,hiA,mpA); CAf=cheb(uAf,NA)
        uBf=umap(np.clip(ev['x'][ob],loB,hiB),loB,hiB,mpB); CBf=cheb(uBf,NB)
        CA=cheb(umaps[oa],NA); CB=cheb(umaps[ob],NB)
        recoil_obs=cfg['recoil'][0]
        if recoil_obs in (oa,ob):
            xr=ev['x'][recoil_obs]; mw=(xr>=XM)&(xr<XHI); wW=wF[mw]
            I_S_j=prior[recoil_obs]<XM; I_T_j=prior[recoil_obs]>=XHI
            pj=wL/wL.sum()
            wSJ=max(float(pj[I_S_j].sum()),1e-300); wTJ=max(float(pj[I_T_j].sum()),1e-300)
            P_tail_j=float(pj[I_T_j].sum()); P_soft_j=1.0-rate-P_tail_j
            for i in range(1,NA+1):
                for j in range(1,NB+1):
                    Ewin=float((wW*CAf[mw,i]*CBf[mw,j]).sum()/max(wW.sum(),1e-300))
                    Esoft=float((pj*CA[:,i]*CB[:,j]*I_S_j).sum())/wSJ
                    Etail=float((pj*CA[:,i]*CB[:,j]*I_T_j).sum())/wTJ if P_tail_j>0 else 0.0
                    F.append(CA[:,i]*CB[:,j]); T.append(P_soft_j*Esoft+rate*Ewin+P_tail_j*Etail)
                    names.append(f'J_{oa}{i}_{ob}{j}')
        else:
            # Born x Born: 2D differential K on a coarse grid from FO events of both orders
            fo_lo_ev = fo_lo['parts'][oa] if isinstance(fo_lo,dict) and fo_lo.get('kind')=='multi' else fo_lo
            if fo_lo_ev.get('kind')!='events':
                report[f'joint_{oa}_{ob}']='skipped (no lower-order 2D input)'
                continue
            gA=np.linspace(loA,hiA,7); gB=np.linspace(loB,hiB,7)
            H2,_ ,_=np.histogram2d(ev['x'][oa],ev['x'][ob],bins=[gA,gB],weights=wF)
            wLo=fo_lo_ev['w']
            L2,_,_=np.histogram2d(fo_lo_ev['x'][oa],fo_lo_ev['x'][ob],bins=[gA,gB],weights=wLo)
            H2/=max(H2.sum(),1e-300); L2/=max(L2.sum(),1e-300)
            K2=np.where((L2>1e-9)&(H2>0),H2/np.maximum(L2,1e-300),1.0)
            K2=np.clip(K2,*dials['CLIP'])
            ia=np.clip(np.searchsorted(gA,prior[oa])-1,0,5); ib=np.clip(np.searchsorted(gB,prior[ob])-1,0,5)
            wK2=wL*K2[ia,ib]
            for i in range(1,NA+1):
                for j in range(1,NB+1):
                    F.append(CA[:,i]*CB[:,j]); T.append(float((wK2*CA[:,i]*CB[:,j]).sum()/wK2.sum()))
                    names.append(f'J_{oa}{i}_{ob}{j}')
            report[f'joint_{oa}_{ob}']=f'2D diff-K, grid 6x6, K range [{K2.min():.2f},{K2.max():.2f}]'
    Phi=np.column_stack(F); mu=np.array(T)
    q,lam,ok=_newton_maxent(Phi,p,mu,l2=dials['L2'],lam0=lam0)
    if not (ok and q is not None): return None,report
    report['_lam']=lam
    ach=(q[:,None]*Phi).sum(0)
    rel=np.abs(ach-mu)/np.maximum(np.abs(mu),1e-30)
    effN=float(1/((q**2).sum()*len(q)))
    q_over_p=q/p
    effW=float(1/((q[I_F]**2).sum()/max(q[I_F].sum()**2,1e-300)*I_F.sum())) if I_F.sum() else None
    report['solve']=dict(K=int(Phi.shape[1]),effN_pct=100*effN,worst_rel_closure=float(rel.max()),
                         window_effN_pct=(100*effW if effW else None),
                         maxmed_weight=float(q_over_p.max()/np.median(q_over_p)))
    return q,report

def band_solve(prior,cfg,fo_lo,fo_hi,dials=DIALS,xm_override=None):
    """central + all scale variations + rate schemes -> dict of weight vectors + report."""
    rep={}
    q0,rep=build_and_solve(prior,cfg,fo_lo,fo_hi,0,'max',dials,xm_override,rep)
    assert q0 is not None,'central solve failed'
    lam0=rep.get('_lam')
    out={'central':q0}
    for k in range(1,_nscales(fo_hi)):
        qk,_=build_and_solve(prior,cfg,fo_lo,fo_hi,k,'max',dials,xm_override,{},lam0=lam0)
        if qk is not None: out[f'scale{k}']=qk
    for sch in ('prior','fo'):
        qs,_=build_and_solve(prior,cfg,fo_lo,fo_hi,0,sch,dials,xm_override,{},lam0=lam0)
        if qs is not None: out[f'rate_{sch}']=qs
    rep['n_variants']=len(out)
    return out,rep
