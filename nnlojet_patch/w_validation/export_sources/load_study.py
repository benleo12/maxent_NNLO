"""Portable reader for the independent W prediction study's public inputs.

Return moments/config accepted by maxent_upgrade.upgrade. Joint models use
include_rate=False, since the angular tower already fixes that same window rate.
The fixed-order targets and held-out validation predictions are separate files.
"""
from pathlib import Path
import copy,json
import numpy as np

def load_study(charge,model='selected',scale=0,raw=None,directory=None):
    directory=Path(__file__).resolve().parent if directory is None else Path(directory)
    table=json.loads((directory/('w_%s_training.json'%charge.lower())).read_text())
    if model=='selected':model=table['selected']
    config=copy.deepcopy(table['configs'][model])
    idx={x:i for i,x in enumerate(table['accumulator_labels'])}
    allraw=np.array(table['accumulator_mean_pb']);cov=np.array(table['accumulator_covariance_pb2'])
    value=allraw[:,scale] if raw is None else np.asarray(raw,float)
    if value.shape!=(len(idx),):raise ValueError('raw must be one complete accumulator vector')
    def ratios(tags,denominator):
        ii=np.array([idx[x] for x in tags]);d=idx[denominator]
        vals=value[ii]/value[d]
        j=np.zeros((len(ii),len(idx)));j[np.arange(len(ii)),ii]=1/value[d]
        j[:,d]-=value[ii]/value[d]**2
        stat=np.sqrt(np.maximum(np.diag(j@cov@j.T),0))
        scale_values=allraw[ii]/allraw[d]
        se=.5*np.ptp(scale_values,axis=1)
        return vals,stat,se
    moments={'born':{},'recoil':{}}
    for observable,prefix in [('m_w','prof_mw_'),('y_abs','prof_absyw_')]:
        vals,st,sc=ratios([prefix+str(n) for n in range(1,13)],'norm_born')
        moments['born'][observable]=dict(values=vals.tolist(),stat_errors=st.tolist(),
            scale_errors=sc.tolist(),errors=np.hypot(st,sc).tolist())
    specs=[('pt_w','prof_wptw_',20,25.,500.,25.)]
    if model!='B':specs.append(('ctw1','prof_wctw_',6,.3,2.,0.))
    for observable,prefix,degree,xmatch,xhi,soft in specs:
        vals,st,sc=ratios([prefix+str(n) for n in range(1,degree+1)],prefix+'0')
        rate,rst,rsc=ratios([prefix+'0'],'norm_born')
        moments['recoil'][observable]=dict(window_values=vals.tolist(),stat_errors=st.tolist(),
            scale_errors=sc.tolist(),window_errors=np.hypot(st,sc).tolist(),rate=float(rate[0]),
            rate_stat=float(rst[0]),rate_scale=float(rsc[0]),x_match=xmatch,x_hi=xhi,soft_lo=soft)
    degree={'B':0,'A':0,'J2':2,'J4':4}[model]
    if degree:
        pairs=[(k,n) for k in range(1,degree+1) for n in range(degree+1)]
        vals,st,sc=ratios(['prof_yc_%d%d'%pair for pair in pairs],'prof_wctw_0')
        rate,rst,rsc=ratios(['prof_wctw_0'],'norm_born')
        moments['mixed']={'rapidity_angle':dict(values=dict(zip(pairs,vals)),stat_errors=dict(zip(pairs,st)),
            errors=dict(zip(pairs,np.hypot(st,sc))),rate=float(rate[0]),rate_stat=float(rst[0]),rate_scale=float(rsc[0]))}
    return moments,config,float(value[idx['norm_born']])

def normalized_covariance(table):
    """Full covariance of normalized raw moments, before selecting constraints."""
    mean=np.array(table['accumulator_mean_pb'])[:,0];cov=np.array(table['accumulator_covariance_pb2'])
    j=np.eye(len(mean))/mean[0];j[:,0]-=mean/mean[0]**2
    return j@cov@j.T
