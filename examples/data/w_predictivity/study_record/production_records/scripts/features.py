"""Direct Cartesian CS features for the declared W study (massless convention)."""
import numpy as np
from make_booking import definitions

def smoother(t):
    t=np.clip(t,0,1)
    return t*t*t*(t*(6*t-15)+10)

def profile(x,a,b,c,d):
    lx=np.log(np.maximum(x,1e-30))
    up=smoother((lx-np.log(a))/(np.log(b)-np.log(a))) if b>a else (x>=a).astype(float)
    dn=1-smoother((lx-np.log(c))/(np.log(d)-np.log(c))) if d>c else (x<c).astype(float)
    return up*dn

def coordinates(l,nu):
    if not np.isfinite(l).all() or not np.isfinite(nu).all(): raise ValueError('Nonfinite lepton/neutrino four-vector')
    v=l+nu;mass=np.sqrt(np.maximum(v[:,3]**2-np.sum(v[:,:3]**2,axis=1),0))
    qt=np.hypot(v[:,0],v[:,1]);mt=np.hypot(mass,qt)
    sg=np.where(v[:,2]<0,-1.,1.)
    c=sg*((l[:,3]+l[:,2])*(nu[:,3]-nu[:,2])-(l[:,3]-l[:,2])*(nu[:,3]+nu[:,2]))/(mass*mt)
    c=np.clip(c,-1,1); y=np.arctanh(v[:,2]/v[:,3])
    dot=l[:,0]*v[:,0]+l[:,1]*v[:,1]
    cross=(-l[:,0]*v[:,1]+l[:,1]*v[:,0])*sg
    ez=(l[:,3]*v[:,3]-l[:,2]*v[:,2])/mt
    rho=qt/mt;ax=2*(mt*dot-qt*qt*ez)/(mass*mass*mt);ay=2*cross/(mass*mt)
    sx=np.divide(ax,rho,out=np.zeros_like(ax),where=qt>1e-12)
    sy=np.divide(ay,rho,out=np.zeros_like(ay),where=qt>1e-12)
    ptl=np.hypot(l[:,0],l[:,1]);ptnu=np.hypot(nu[:,0],nu[:,1])
    mtlep=np.sqrt(np.maximum(2*(ptl*ptnu-l[:,0]*nu[:,0]-l[:,1]*nu[:,1]),0))
    eta=np.arctanh(l[:,2]/l[:,3])
    return dict(mass=mass,qt=qt,y=y,c=c,sx=sx,sy=sy,ax=ax,ay=ay,rho=rho,ptl=ptl,ptnu=ptnu,mtlep=mtlep,eta=eta)

def features(l,nu,roles=None):
    e=coordinates(l,nu);m=e['mass'];c=e['c'];qt=e['qt']
    ct=np.polynomial.chebyshev.chebvander(c,10)
    yt=np.polynomial.chebyshev.chebvander(np.clip(np.abs(e['y'])/2-1,-1,1),4)
    lo=np.arctan((40**2-80.385**2)/(80.385*2.085));hi=np.arctan((1000**2-80.385**2)/(80.385*2.085))
    um=2*(np.arctan((m*m-80.385**2)/(80.385*2.085))-lo)/(hi-lo)-1
    mt=np.polynomial.chebyshev.chebvander(np.clip(um,-1,1),2)
    wc=profile(c+1,.3,.4,1.6,1.7)
    bornpt=.5*m*np.sqrt(np.maximum(1-c*c,0))
    gate=smoother(np.log(np.maximum(bornpt,1e-30)/25)/np.log(28/25))
    windows=dict(N=gate*profile(c+1,.4,.5,1.5,1.6),G=gate*wc,W=gate*profile(c+1,.25,.35,1.65,1.75))
    p2=.5*(1-3*c*c);rho=e['rho'];sx=e['sx'];sy=e['sy'];ax=e['ax'];ay=e['ay']
    hard=np.column_stack([p2,c,2*c*sx,sx*sx-sy*sy,sx])*profile(np.maximum(qt,1e-12),25,50,500,500)[:,None]
    soft=np.column_stack([rho*rho*p2,rho*rho*c,2*c*ax,ax*ax-ay*ay,ax])
    iq=np.where(qt>=50,2,np.where(qt>=25,1,0));ia=np.where(abs(c)>=.7,2,np.where(abs(c)>.6,1,0))
    im=np.where(m>=84,2,np.where(m>=76,1,0))
    out=[];names=[]
    for row in definitions():
        if roles is not None and row['role'] not in roles: continue
        z=row['metadata'];block=z.get('block')
        if block=='D': value=wc*ct[:,z['n']]
        elif block=='M': value=wc*mt[:,z['i']]*ct[:,z['j']]
        elif block in ['N','G','W']: value=windows[block]*ct[:,z['n']]*(yt[:,z['k']] if 'k' in z else 1)
        elif block in ['H','S']: value=(hard if block=='H' else soft)[:,z['h']]*yt[:,z['k']]
        elif row['role']=='exploratory':
            keep=(iq==z['iq'])&((ia if z['category']=='qc' else im)==z['region'])
            if 'projected_cut' in z: keep&=bornpt>z['projected_cut']
            elif z['cut']: keep&=e['ptl']>z['cut']
            value=keep.astype(float)
        else:
            key=z['selection']
            if key=='pt28': keep=e['ptl']>28
            elif key=='pt32': keep=e['ptl']>32
            elif key=='hard': keep=qt>50
            elif key=='pt32hard': keep=(e['ptl']>32)&(qt>50)
            elif key=='mt60': keep=e['mtlep']>60
            else: keep=(e['mtlep']>60)&(e['ptnu']>30)&(abs(e['eta'])<1.7)
            value=keep.astype(float)
        out.append(value);names.append(row['name'])
    return np.column_stack(out),names,e
