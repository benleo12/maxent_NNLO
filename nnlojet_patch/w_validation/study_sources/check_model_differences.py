"""Independent finite-difference check of paired asymmetry propagation."""
from pathlib import Path
import json,numpy as np
rng=np.random.default_rng(9138);n=250;bins=11
mp=rng.uniform(80,180,bins);mn=rng.uniform(50,120,bins);ap=mp*rng.uniform(.96,1.04,bins);an=mn*rng.uniform(.96,1.04,bins)
ip=rng.normal(size=(n,bins));inn=rng.normal(size=(n,bins));iap=.8*ip+.2*rng.normal(size=(n,bins));ian=.8*inn+.2*rng.normal(size=(n,bins))
jp=2*mn/(mp+mn)**2;jn=-2*mp/(mp+mn)**2;jap=2*an/(ap+an)**2;jan=-2*ap/(ap+an)**2
def value(t,k):
 p=mp+t*ip[k];m=mn+t*inn[k];a=ap+t*iap[k];b=an+t*ian[k]
 return (p-m)/(p+m)-(a-b)/(a+b)
eps=1e-3;fd=np.stack([(value(eps,k)-value(-eps,k))/(2*eps) for k in range(n)])
exact=ip*jp+inn*jn-iap*jap-ian*jan
relative=float(np.max(np.abs(fd-exact))/np.max(np.abs(exact)));assert relative<1e-7
# Cross-model formula per charge versus direct paired influence covariance.
cross=np.zeros((bins,bins));direct=np.zeros_like(cross)
for im,ia,jm,ja in [(ip,iap,jp,jap),(inn,ian,jn,jan)]:
 cm=im.T@im;ca=ia.T@ia;cma=im.T@ia
 cross+=cm*np.outer(jm,jm)+ca*np.outer(ja,ja)-cma*np.outer(jm,ja)-cma.T*np.outer(ja,jm)
 d=im*jm-ia*ja;direct+=d.T@d
covrel=float(np.max(np.abs(cross-direct))/np.max(np.abs(direct)));assert covrel<1e-12
result=dict(finite_difference_relative_max_norm=relative,cross_covariance_vs_paired_influence_relative_max_norm=covrel,production_or_validation_data_used=False)
(Path(__file__).resolve().parents[1]/'output/MODEL_DIFFERENCE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
