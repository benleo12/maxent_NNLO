"""Create the exact common-vector bookings and an isolated NNLOJET patch."""
from pathlib import Path
import argparse, hashlib, json, re, shutil

def definitions():
    rows=[]
    def add(name,expr,role,meta):
        rows.append(dict(name=name,expression=expr,role=role,metadata=meta))
    for n in range(7,11): add('pred_D_%d'%n,'wc*cheb_T(c,%d)'%n,'fit',dict(block='D',n=n))
    for i in range(1,3):
        for j in range(3): add('pred_M_%d_%d'%(i,j),'wc*eval_chebT_mw(npar,%d)*cheb_T(c,%d)'%(i,j),'fit',dict(block='M',i=i,j=j))
    for block,index in [('N',1),('G',2),('W',3)]:
        for n in range(7): add('pred_%s_%d'%(block,n),'wg(%d)*cheb_T(c,%d)'%(index,n),'fit',dict(block=block,n=n))
        for k in range(1,3):
            for n in range(3): add('pred_%sJ_%d_%d'%(block,k,n),'wg(%d)*cheb_T(uy,%d)*cheb_T(c,%d)'%(index,k,n),'fit',dict(block=block,k=k,n=n))
    for block,letter in [('H','hh'),('S','hs')]:
        for k in range(3):
            for h in range(5): add('pred_%s_%d_%d'%(block,k,h),'%s(%d)*cheb_T(uy,%d)'%(letter,h+1,k),'fit',dict(block=block,k=k,h=h))
    for category,field in [('qc','ia'),('qm','im')]:
        for iq in range(3):
            for region in range(3):
                cond='iq==%d.and.%s==%d'%(iq,field,region)
                for cut in [0,27,30]:
                    extra='' if cut==0 else '.and.ptl>%d.0_wp'%cut
                    add('expl_%s_%d_%d_pt%d'%(category,iq,region,cut),'merge(1.0_wp,0.0_wp,%s%s)'%(cond,extra),'exploratory',dict(category=category,iq=iq,region=region,cut=cut))
                if category=='qc':
                    for cut in [27,30]: add('expl_qc_%d_%d_proj%d'%(iq,region,cut),'merge(1.0_wp,0.0_wp,%s.and.bornpt>%d.0_wp)'%(cond,cut),'exploratory',dict(category=category,iq=iq,region=region,projected_cut=cut))
    for name,cond in [('pt28','ptl>28.0_wp'),('pt32','ptl>32.0_wp'),('hard','qt>50.0_wp'),('pt32hard','ptl>32.0_wp.and.qt>50.0_wp'),('mt60','mtlep>60.0_wp'),('combined','mtlep>60.0_wp.and.ptnu>30.0_wp.and.abs(v1_y(pl))<1.7_wp')]:
        add('hold_'+name,'merge(1.0_wp,0.0_wp,%s)'%cond,'validation',dict(selection=name))
    for i,row in enumerate(rows,1): row['code']=i
    return rows

HEAD='''  ! W predictivity functions: direct kinematics, no memoized dependencies.
  real(kind=wp) function eval_wpred(npar,code) result(res)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    integer, intent(in) :: npar,code
    integer :: ilep, key, iq, ia, im
    real(kind=wp), dimension(4) :: pl,pn,pv
    real(kind=wp), dimension(3) :: wg
    real(kind=wp), dimension(5) :: hh,hs
    real(kind=wp) :: mass,qt,mtv,c,uy,wc,gate,bornpt,tt,sgn
    real(kind=wp) :: rho,ax,ay,sx,sy,ez,dot,cross,ptl,ptnu,mtlep,p2,wh
    logical,save :: cached_valid=.false.
    real(kind=wp),save :: cached_pl(4),cached_pn(4),cached_values(@NFEATURES@)
    res=0.0_wp
    if(npar<2) return
    ilep=abs(code)/1000; key=mod(abs(code),1000)
    pl=kin(npar)%p(:,npar-2+ilep)
    pn=kin(npar)%p(:,npar+1-ilep)
    if(key<1.or.key>@NFEATURES@) return
    if(cached_valid) then
      if(all(pl==cached_pl).and.all(pn==cached_pn)) then
        res=cached_values(key)
        return
      endif
    endif
    ! Exact equality with a previously finite key already proves finiteness.
    ! New keys are checked explicitly before any kinematic arithmetic.
    if(.not.all(ieee_is_finite(pl)).or..not.all(ieee_is_finite(pn))) then
      cached_valid=.false.
      error stop 'nonfinite four-vector in W predictivity booking'
    endif
    if(key>=@FIRST_HOLD@) then
      pv=pl+pn;qt=v1_pt(pv)
      ptl=v1_pt(pl);ptnu=v1_pt(pn)
      mtlep=sqrt(max(2.0_wp*(ptl*ptnu-pl(1)*pn(1)-pl(2)*pn(2)),0.0_wp))
      select case(key)
@FASTCASES@
      end select
      return
    endif
    pv=pl+pn; mass=v1_minv(pv); qt=v1_pt(pv); mtv=sqrt(mass*mass+qt*qt)
    if(mass<=0.0_wp.or.mtv<=0.0_wp) return
    sgn=1.0_wp; if(pv(3)<0.0_wp) sgn=-1.0_wp
    c=sgn*((pl(4)+pl(3))*(pn(4)-pn(3))-(pl(4)-pl(3))*(pn(4)+pn(3)))/(mass*mtv)
    c=max(-1.0_wp,min(1.0_wp,c))
    uy=abs(v1_y(pv))/2.0_wp-1.0_wp
    wc=prof_w(c+1.0_wp,0.3_wp,0.4_wp,1.6_wp,1.7_wp)
    bornpt=0.5_wp*mass*sqrt(max(1.0_wp-c*c,0.0_wp))
    gate=0.0_wp
    if(bornpt>25.0_wp) then
      tt=min(log(bornpt/25.0_wp)/log(28.0_wp/25.0_wp),1.0_wp)
      gate=6.0_wp*tt**5-15.0_wp*tt**4+10.0_wp*tt**3
    endif
    wg(1)=gate*prof_w(c+1.0_wp,0.4_wp,0.5_wp,1.5_wp,1.6_wp)
    wg(2)=gate*wc
    wg(3)=gate*prof_w(c+1.0_wp,0.25_wp,0.35_wp,1.65_wp,1.75_wp)
    rho=qt/mtv
    dot=pl(1)*pv(1)+pl(2)*pv(2)
    cross=(-pl(1)*pv(2)+pl(2)*pv(1))*sgn
    ez=(pl(4)*pv(4)-pl(3)*pv(3))/mtv
    ! ax=rho*sx and ay=rho*sy remain finite and vanish at qT=0.
    ax=2.0_wp*(mtv*dot-qt*qt*ez)/(mass*mass*mtv)
    ay=2.0_wp*cross/(mass*mtv)
    sx=0.0_wp; sy=0.0_wp
    if(qt>1.0e-12_wp) then
      sx=ax/rho; sy=ay/rho
    endif
    p2=0.5_wp*(1.0_wp-3.0_wp*c*c)
    wh=prof_w(max(qt,1.0e-12_wp),25.0_wp,50.0_wp,500.0_wp,500.0_wp)
    hh=wh*[p2,c,2.0_wp*c*sx,sx*sx-sy*sy,sx]
    hs=[rho*rho*p2,rho*rho*c,2.0_wp*c*ax,ax*ax-ay*ay,ax]
    ptl=v1_pt(pl); ptnu=v1_pt(pn)
    mtlep=sqrt(max(2.0_wp*(ptl*ptnu-pl(1)*pn(1)-pl(2)*pn(2)),0.0_wp))
    iq=0; if(qt>=25.0_wp) iq=1; if(qt>=50.0_wp) iq=2
    ia=0; if(abs(c)>0.6_wp) ia=1; if(abs(c)>=0.7_wp) ia=2
    im=0; if(mass>=76.0_wp) im=1; if(mass>=84.0_wp) im=2
'''
TAIL='''    cached_pl=pl; cached_pn=pn; cached_valid=.true.
    res=cached_values(key)
  end function eval_wpred

  real(kind=wp) function eval_wpred_eta(npar,code) result(res)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    integer, intent(in) :: npar,code
    integer :: ilep,key
    real(kind=wp), dimension(4) :: pl,pn
    real(kind=wp) :: ptl,ptnu,mtlep
    res=-1.0_wp
    if(npar<2) return
    ilep=abs(code)/1000; key=mod(abs(code),1000)
    pl=kin(npar)%p(:,npar-2+ilep); pn=kin(npar)%p(:,npar+1-ilep)
    if(.not.all(ieee_is_finite(pl)).or..not.all(ieee_is_finite(pn))) error stop 'nonfinite four-vector in W eta booking'
    ptl=v1_pt(pl); ptnu=v1_pt(pn)
    mtlep=sqrt(max(2.0_wp*(ptl*ptnu-pl(1)*pn(1)-pl(2)*pn(2)),0.0_wp))
    if(key==1.and.ptl<=28.0_wp) return
    if(key==2.and.ptl<=32.0_wp) return
    if(key==3.and.mtlep<=60.0_wp) return
    res=abs(v1_y(pl))
  end function eval_wpred_eta

  real(kind=wp) function eval_wpred_region(npar,code) result(res)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    integer, intent(in) :: npar,code
    integer :: ilep,key,iq,ia,im
    real(kind=wp), dimension(4) :: pl,pn,pv
    real(kind=wp) :: mass,qt,mtv,c,bornpt,ptl
    res=-1.0_wp
    if(npar<2) return
    ilep=abs(code)/1000; key=mod(abs(code),1000)
    pl=kin(npar)%p(:,npar-2+ilep); pn=kin(npar)%p(:,npar+1-ilep)
    if(.not.all(ieee_is_finite(pl)).or..not.all(ieee_is_finite(pn))) error stop 'nonfinite four-vector in W region booking'
    pv=pl+pn;mass=v1_minv(pv);qt=v1_pt(pv);mtv=sqrt(mass*mass+qt*qt)
    if(mass<=0.0_wp) return
    c=((pl(4)+pl(3))*(pn(4)-pn(3))-(pl(4)-pl(3))*(pn(4)+pn(3)))/(mass*mtv)
    c=max(-1.0_wp,min(1.0_wp,c))
    bornpt=0.5_wp*mass*sqrt(max(1.0_wp-c*c,0.0_wp));ptl=v1_pt(pl)
    if((key==2.or.key==7).and.ptl<=27.0_wp) return
    if((key==3.or.key==8).and.ptl<=30.0_wp) return
    if(key==4.and.bornpt<=27.0_wp) return
    if(key==5.and.bornpt<=30.0_wp) return
    iq=0;if(qt>=25.0_wp) iq=1;if(qt>=50.0_wp) iq=2
    ia=0;if(abs(c)>0.6_wp) ia=1;if(abs(c)>=0.7_wp) ia=2
    im=0;if(mass>=76.0_wp) im=1;if(mass>=84.0_wp) im=2
    if(key<=5) then
      res=real(3*iq+ia,wp)+0.5_wp
    else
      res=real(3*iq+im,wp)+0.5_wp
    endif
  end function eval_wpred_region
'''

def source_body(rows):
    selected=[r for r in rows if r['role']=='validation']
    fast='\n'.join('      case(%d)\n        res=%s'%(r['code'],r['expression']) for r in selected)
    head=HEAD.replace('@NFEATURES@',str(len(rows))).replace('@FIRST_HOLD@',str(min(r['code'] for r in selected))).replace('@FASTCASES@',fast)
    return head+''.join('    cached_values(%d)=%s\n'%(r['code'],r['expression']) for r in rows)+TAIL

def install(root,old):
    rows=definitions(); out=root/'booking';out.mkdir(parents=True,exist_ok=True)
    (out/'features.json').write_text(json.dumps(rows,indent=2)+'\n')
    src=source_body(rows)
    (out/'w_predictivity_functions.f90').write_text(src)
    core=root/'source/driver/core';ef=core/'EvalFuncs.f90';ob=core/'Observables.f90'
    e=ef.read_text();o=ob.read_text()
    assert 'function eval_wpred(' not in e,'Do not patch an already patched source'
    assert 'function eval_wval_yctw' in e
    e=e.replace('end module EvalFuncs_mod',src+'\nend module EvalFuncs_mod')
    for ilep in [1,2]:
        marker='      call bind_obs("costh_w", "signed cos th*_CS of the charged lepton (W rest frame)", EvalObs_t(eval_mem=eval_costhCS_w, imem=-%d))'%ilep
        assert o.count(marker)==1
        extra=['      call bind_obs("%s", "W predictivity common-vector function", &\n                    EvalObs_t(eval_mem=eval_wpred, imem=-%d))'%(r['name'],ilep*1000+r['code']) for r in rows]
        extra+=['      call bind_obs("pred_eta%d", "W held-out lepton eta", &\n                    EvalObs_t(eval_mem=eval_wpred_eta, imem=-%d))'%(k,ilep*1000+k) for k in [1,2,3]]
        extra+=['      call bind_obs("pred_region%d", "W disjoint diagnostic region", &\n                    EvalObs_t(eval_mem=eval_wpred_region, imem=-%d))'%(k,ilep*1000+k) for k in range(1,9)]
        o=o.replace(marker,marker+'\n'+'\n'.join(extra))
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ef,ob]}
    ef.write_text(e);ob.write_text(o)
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ef,ob]}
    (out/'patch_provenance.json').write_text(json.dumps(dict(before=before,after=after),indent=2)+'\n')
    for charge in ['WP','WM']:
        for group in ['T','V']:
            for channel in ['LO','R','V','RR','RV','VV']:
                origin=old/'cards'/charge/group/channel
                dest=root/'cards'/charge/group/channel;dest.mkdir(parents=True,exist_ok=True)
                card=(origin/'rc.run').read_text().replace('W_VALID_'+group,'W_PRED_'+group)
                tags=[r for r in rows if (r['role']=='fit' if group=='T' else r['role']=='validation')]
                lines=['  %s > %s min=-1.001 max=1.001 nbins=1 profile=T'%(r['name'],r['name']) for r in tags]
                if group=='V':
                    edges='[0.0,0.21,0.42,0.63,0.84,1.05,1.37,1.52,1.74,1.95,2.18,2.5]'
                    lines=['  chebT_mw_1 > norm_born min=-1.001 max=1.001 nbins=1']+lines
                    lines+=['  pred_eta%d > %s %s'%(k,name,edges) for k,name in [(1,'eta28'),(2,'eta32'),(3,'eta_mt60')]]
                    lines+=['  pred_region%d > region%d min=0 max=9 nbins=9'%(k,k) for k in range(1,9)]
                    card=re.sub(r'HISTOGRAMS\n.*?END_HISTOGRAMS','HISTOGRAMS\n'+'\n'.join(lines)+'\nEND_HISTOGRAMS',card,flags=re.S)
                else: card=card.replace('END_HISTOGRAMS','\n'.join(lines)+'\nEND_HISTOGRAMS')
                (dest/'rc.run').write_text(card)
                (dest/'pilot.run').write_text(re.sub(r'production\s*=.*','production = 1000[2]',card))
                oldname='%s.W_VALID_%s.y1.00E-07.%s'%(charge,group,channel)
                newname='%s.W_PRED_%s.y1.00E-07.%s'%(charge,group,channel)
                shutil.copy2(origin/oldname,dest/newname)
    print(json.dumps(dict(new_fit_functions=sum(r['role']=='fit' for r in rows),exploratory=sum(r['role']=='exploratory' for r in rows),validation_scalar=sum(r['role']=='validation' for r in rows),source_hashes=after)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('old',type=Path);ap.add_argument('--cache-upgrade',action='store_true');a=ap.parse_args()
    if a.cache_upgrade:
        out=a.root/'booking';oldsrc=(out/'w_predictivity_functions.f90').read_text()
        rows=definitions();src=source_body(rows)
        assert oldsrc!=src,'Requested cache source is already installed'
        ef=a.root/'source/driver/core/EvalFuncs.f90';e=ef.read_text();assert e.count(oldsrc)==1
        archive=out/('pre_cache' if 'cached_values' not in oldsrc else ('pre_nonfinite_guard' if 'ieee_is_finite' not in oldsrc else ('pre_fastpath' if not (out/'pre_fastpath').exists() else ('pre_cut_fastpath' if not (out/'pre_cut_fastpath').exists() else 'pre_cut_fastpath_fix'))));archive.mkdir()
        if (out/'cache_upgrade.json').exists():shutil.copy2(out/'cache_upgrade.json',archive/'cache_upgrade.json')
        shutil.copy2(out/'w_predictivity_functions.f90',archive/'w_predictivity_functions.f90')
        shutil.copy2(ef,archive/'EvalFuncs.f90')
        ef.write_text(e.replace(oldsrc,src));(out/'w_predictivity_functions.f90').write_text(src)
        obs=a.root/'source/driver/core/Observables.f90';o=obs.read_text()
        if '"pred_region1"' not in o:
            for ilep in [1,2]:
                marker='      call bind_obs("costh_w", "signed cos th*_CS of the charged lepton (W rest frame)", EvalObs_t(eval_mem=eval_costhCS_w, imem=-%d))'%ilep
                extra=['      call bind_obs("pred_region%d", "W disjoint diagnostic region", &\n                    EvalObs_t(eval_mem=eval_wpred_region, imem=-%d))'%(k,ilep*1000+k) for k in range(1,9)]
                assert o.count(marker)==1;o=o.replace(marker,marker+'\n'+'\n'.join(extra))
            shutil.copy2(obs,archive/'Observables.f90');obs.write_text(o)
        (out/'cache_upgrade.json').write_text(json.dumps(dict(before=hashlib.sha256(e.encode()).hexdigest(),after=hashlib.sha256(ef.read_bytes()).hexdigest(),key='Exact equality of all eight charged-lepton/neutrino four-vector components; fixed constants; one thread per process.'),indent=2)+'\n')
        print('Content-keyed cache installed; old function bodies preserved')
    else: install(a.root,a.old)
