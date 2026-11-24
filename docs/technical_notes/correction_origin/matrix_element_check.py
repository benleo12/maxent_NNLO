"""Algebra check of a published QCD matrix-element/shower ratio.

This is not a simulation of the paper's shower. Equations (8) and (13) of
Miu and Sjostrand, hep-ph/9812455, concern the single-emission expansion of
the shower specified there, with fixed couplings/PDF conventions. Their
common collinear poles cancel. Hadronic averaging, multiple emissions,
fiducial cuts, and higher orders are not represented by this calculation.
"""
from pathlib import Path
import json
import numpy as np
from scipy.fft import dctn

root=Path(__file__).resolve().parent
n=128
x=np.cos(np.pi*(np.arange(n)+.5)/n)
z=(x[:,None]+1)/2
c=x[None,:]
t=-(1-z)*(1-c)/2
u=-(1-z)*(1+c)/2
qq=(t*t+u*u+2*z)/(1+z*z)
qg=(1+u*u+2*z*t)/(1+2*z*(t+u))
assert np.max(abs(qq-(1-(1-z)**2/(2*(1+z*z))*(1-c*c))))<1e-14
assert np.max(abs(qg-(1+u*(u-2*z)/((1-z)**2+z*z))))<1e-14

xx=np.linspace(-1,1,801)
zz=(xx[:,None]+1)/2
cc=xx[None,:]
tt=-(1-zz)*(1-cc)/2
uu=-(1-zz)*(1+cc)/2
true=[(tt*tt+uu*uu+2*zz)/(1+zz*zz),
      (1+uu*uu+2*zz*tt)/(1+2*zz*(tt+uu))]
V=np.polynomial.chebyshev.chebvander(xx,12)
rows=[]
for name,ratio,exact in zip(('qqbar','qg'),(qq,qg),true):
    coefficients=dctn(np.log(ratio),type=2)/(n*n)
    coefficients[0,:]/=2
    coefficients[:,0]/=2
    for d in (1,2,3,4,5,6,8,10,12):
        logapprox=V[:,:d+1]@coefficients[:d+1,:d+1]@V[:,:d+1].T
        err=float(np.max(abs(np.exp(logapprox)/exact-1)))
        rows.append(dict(channel=name,degree=d,
            nonconstant_terms=((d+1)*(d//2+1)-1 if name=='qqbar' else (d+1)**2-1),
            maximum_relative_error_on_grid=err))

# Exact angular expansion at fixed partonic energy in the hardest qqbar limit.
# R=(1+c^2)/2=(3+T2(c))/4. These are coefficients of LOG R, as needed
# for the exponential correction, not coefficients of R itself.
q=3-np.sqrt(8)
constant=np.log((.75+np.sqrt(.75**2-.25**2))/2)
angular=[]
for k in range(1,5):
    co=np.zeros(2*k+1); co[0]=constant
    for j in range(1,k+1): co[2*j]=2*(-1)**(j+1)*q**j/j
    f=np.exp(np.polynomial.chebyshev.chebval(xx,co))
    error=float(np.max(abs(f/((1+xx*xx)/2)-1)))
    tail_bound=2*q**(k+1)/((k+1)*(1-q))
    angular.append(dict(nonconstant_even_modes=k,maximum_degree=2*k,
        maximum_relative_error_on_grid=error,
        uniform_relative_error_bound=float(np.expm1(tail_bound))))
result=dict(source='https://arxiv.org/abs/hep-ph/9812455',equations=[8,13],
    scope=__doc__,variables=dict(z='M^2/s_hat in [0,1]',c='partonic scattering cosine in [-1,1]'),
    two_dimensional_log_ratio_truncations=rows,
    qqbar_hard_limit_angular_log_expansion=angular,
    caveat='An approximation check of an exact partonic ratio, not a finite-statistics moment fit or a universal NNLO claim.')
(root/'matrix_element_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(angular,indent=2))
