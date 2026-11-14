"""Compile exact booked function bodies and compare with event-array features."""
from pathlib import Path
import argparse,json,re,subprocess,hashlib
import numpy as np
from make_booking import definitions
from features import features,coordinates

ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();root=a.root
out=root/'technical';out.mkdir(exist_ok=True)
source=(root/'source/driver/core/EvalFuncs.f90').read_text()
def extract(name):
    pat=r'^  (?:pure |real\(kind=wp\) )?function '+name+r'\(.*?^  end function '+name+r'\s*$'
    match=re.search(pat,source,re.M|re.S)
    assert match,name
    return match.group(0)+'\n'
head='''module check_mod
implicit none
integer,parameter::wp=kind(1d0)
type state
real(wp)::p(4,2)
end type
type(state)::kin(2)
contains
real(wp) function v1_pt(p)
real(wp),intent(in)::p(4)
v1_pt=sqrt(p(1)**2+p(2)**2)
end function
real(wp) function v1_minv(p)
real(wp),intent(in)::p(4)
v1_minv=sqrt(max(p(4)**2-sum(p(1:3)**2),0d0))
end function
real(wp) function v1_y(p)
real(wp),intent(in)::p(4)
v1_y=0.5d0*log((p(4)+p(3))/(p(4)-p(3)))
end function
'''
names=['cheb_T','bw_u','prof_w','eval_chebT_mw','eval_wpred','eval_wpred_eta','eval_wpred_region']
body=head+''.join(extract(n) for n in names)+'end module\n'
nfeat=len(definitions())
body+='''program check
use check_mod
implicit none
integer::i,j,n,ios
real(wp)::pl(4),pn(4),vals(%d+11),tmp(4),early(6)
read(*,*)n
do i=1,n
read(*,*)pl,pn
kin(2)%%p(:,1)=pl;kin(2)%%p(:,2)=pn
do j=1,6
early(j)=eval_wpred(2,1151+j)
enddo
do j=1,%d
vals(j)=eval_wpred(2,1000+j)
enddo
if(any(early/=vals(152:157))) error stop 'cut fast path differs from cached full vector'
do j=1,3
vals(size(vals)-11+j)=eval_wpred_eta(2,1000+j)
enddo
do j=1,8
vals(size(vals)-8+j)=eval_wpred_region(2,1000+j)
enddo
write(*,'(*(ES25.16E3,1X))')vals
kin(2)%%p(:,1)=pn;kin(2)%%p(:,2)=pl
do j=1,%d
vals(j)=eval_wpred(2,2000+j)
enddo
do j=1,3
vals(size(vals)-11+j)=eval_wpred_eta(2,2000+j)
enddo
do j=1,8
vals(size(vals)-8+j)=eval_wpred_region(2,2000+j)
enddo
write(*,'(*(ES25.16E3,1X))')vals
enddo
end program
'''%(nfeat,nfeat,nfeat)
(out/'pointwise.f90').write_text(body)
subprocess.run(['gfortran','-O2','-ffree-line-length-none',str(out/'pointwise.f90'),'-o',str(out/'pointwise')],check=True)
rng=np.random.RandomState(6142026)
random=np.column_stack([rng.uniform(40,160,1600),rng.uniform(0,650,1600),rng.uniform(-5,5,1600),rng.uniform(-.999,.999,1600),rng.uniform(-np.pi,np.pi,1600)])
special=[]
for mass in [39.,40.,49.9,50.,56.,76.,80.385,84.,1000.,1100.]:
 for qt in [0.,1e-10,24.999999,25.,25.000001,49.999999,50.,50.000001,499.999999,500.,500.000001]:
  for c in [-.75000001,-.75,-.70000001,-.7,-.60000001,-.6,-.5,0.,.5,.6,.60000001,.7,.70000001,.75,.75000001]:
   special.append([mass,qt,0.7,c,0.4])
params=np.vstack([random,special]);mass,qt,y,c,phi=params.T
sg=np.where(y<0,-1.,1.);sf=np.sqrt(1-c*c);sx=sf*np.cos(phi);sy=sf*np.sin(phi)*sg;cz=c*sg;mt=np.hypot(mass,qt)
px=.5*(qt+mt*sx);py=.5*mass*sy
el=.5*(mt+qt*sx)*np.cosh(y)+.5*mass*cz*np.sinh(y)
pz=.5*(mt+qt*sx)*np.sinh(y)+.5*mass*cz*np.cosh(y)
l=np.column_stack([px,py,pz,el]);v=np.column_stack([qt,np.zeros_like(qt),mt*np.sinh(y),mt*np.cosh(y)]);nu=v-l
with (out/'pointwise_input.txt').open('w') as f:
 f.write(str(len(l))+'\n');np.savetxt(f,np.column_stack([l,nu]),fmt='%.17e')
with (out/'pointwise_input.txt').open() as fi,(out/'pointwise_output.txt').open('w') as fo:subprocess.run([str(out/'pointwise')],stdin=fi,stdout=fo,check=True)
ref,names,e=features(l,nu);allraw=np.loadtxt(out/'pointwise_output.txt');raw=allraw[:,:nfeat];f1=raw[::2];f2=raw[1::2]
assert np.isfinite(raw).all() and np.isfinite(ref).all()
fit=np.array([r['role']=='fit' for r in definitions()])
delta=np.abs(ref-f1);continuous=float(delta[:,fit].max());charge=float(np.abs(f1-f2).max())
assert continuous<3e-8,continuous
assert charge<3e-8,charge
# Boolean ties exactly on declared edges can differ by floating-point rounding.
bad=np.any(delta[:,~fit]>1e-8,axis=1)
boundary=np.zeros(len(l),bool)
for array,edges in [(e['qt'],[25,50]),(abs(e['c']),[.6,.7]),(e['mass'],[76,84]),(e['ptl'],[27,28,30,32]),(e['mtlep'],[60]),(e['ptnu'],[30]),(abs(e['eta']),[1.7]),(.5*e['mass']*np.sqrt(1-e['c']**2),[27,30])]:
 for edge in edges:boundary|=abs(array-edge)<1e-7
assert not np.any(bad&~boundary),'Boolean disagreement away from declared boundaries'
etas=np.column_stack([np.where(e['ptl']>28,abs(e['eta']),-1),np.where(e['ptl']>32,abs(e['eta']),-1),np.where(e['mtlep']>60,abs(e['eta']),-1)])
eta_delta=np.abs(etas-allraw[::2,nfeat:nfeat+3]);assert np.all(eta_delta[~boundary]<3e-8)
region_def=[('qc','pt0'),('qc','pt27'),('qc','pt30'),('qc','proj27'),('qc','proj30'),('qm','pt0'),('qm','pt27'),('qm','pt30')]
region_ref=[]
for cat,cut in region_def:
 matrix=np.column_stack([ref[:,names.index('expl_%s_%d_%d_%s'%(cat,iq,ir,cut))] for iq in range(3) for ir in range(3)])
 assert np.all(matrix.sum(1)<=1)
 region_ref.append(np.where(matrix.sum(1)>0,matrix@np.arange(9)+.5,-1))
region_ref=np.column_stack(region_ref);region_delta=abs(region_ref-allraw[::2,nfeat+3:])
assert not np.any(region_delta[~boundary]),'Region packing differs away from boundaries'
badmoment=np.column_stack([l[:1],nu[:1]]);badmoment[0,0]=np.nan
from io import StringIO
stream=StringIO();stream.write('1\n');np.savetxt(stream,badmoment,fmt='%.17e')
failure=subprocess.run([str(out/'pointwise')],input=stream.getvalue(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
assert failure.returncode!=0 and 'nonfinite four-vector' in failure.stderr
maximum=float(max(abs(ref).max(),abs(raw).max()));assert maximum<=1.0000001,maximum
summary=dict(points=len(l),features=nfeat,max_continuous_abs_difference=continuous,max_charge_swap_difference=charge,
 max_abs_feature=maximum,max_eta_off_boundary_difference=float(eta_delta[~boundary].max()),boolean_boundary_ties=int(bad.sum()),boolean_off_boundary_mismatches=int((bad&~boundary).sum()),
 region_packing_off_boundary_mismatches=int(np.count_nonzero(region_delta[~boundary])),nonfinite_input_rejected=True,
 source_function_sha256={n:hashlib.sha256(extract(n).encode()).hexdigest() for n in ['eval_wpred','eval_wpred_eta','eval_wpred_region']},
 test='Exact booked bodies compiled with direct kinematic primitives; includes qT=0, clipped maps and both sides of all profile/partition boundaries.')
(out/'POINTWISE_CHECK.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
