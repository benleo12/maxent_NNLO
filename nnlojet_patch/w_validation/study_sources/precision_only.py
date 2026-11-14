"""Expose validation Monte Carlo precision without exposing central predictions."""
from pathlib import Path
import argparse
import json
import numpy as np
from stat_utils import bootstrap_raw

def ratio_cov(m,c,d=0):
    j=np.eye(len(m))/m[d]
    j[:,d]-=m/m[d]**2
    return j@c@j.T

ap=argparse.ArgumentParser()
ap.add_argument('input',type=Path)
ap.add_argument('--charge',choices=['both','WP','WM'],default='both')
args=ap.parse_args()
output={}
means={};covs={};boots={}
passed=[]
for ch in (['WP','WM'] if args.charge=='both' else [args.charge]):
    z=np.load(str(args.input/('V_%s.npz'%ch)),allow_pickle=False)
    m=z['mean'][:,0];c=z['cov'];meta=json.loads(str(z['metadata']))
    means[ch]=m;covs[ch]=c
    boots[ch]=bootstrap_raw(z['acc'],'V',ch)
    item={}
    for tag in ['eta20','eta27','eta30']:
        lo,hi=meta['slices'][tag]
        ind=np.arange(lo,hi)
        total=m[ind].sum()
        a=total/m[0]
        v=np.zeros(len(m));v[ind]=1/m[0];v[0]-=total/m[0]**2
        rb=boots[ch][:,ind].sum(1)/boots[ch][:,0]
        item[tag]=dict(bin_relative_stat=(np.sqrt(np.maximum(np.diag(c)[ind],0))/np.abs(m[ind])).tolist(),
                       acceptance_relative_stat=float(rb.std(ddof=1)/abs(a)),
                       acceptance_relative_stat_delta=float(np.sqrt(max(v@c@v,0))/abs(a)))
        item[tag]['acceptance_target_met']=bool(item[tag]['acceptance_relative_stat']<=.003)
        item[tag]['all_bin_targets_met']=bool(max(item[tag]['bin_relative_stat'])<=.01)
        passed.extend([item[tag]['acceptance_target_met'],item[tag]['all_bin_targets_met']])
    item['runs_per_channel']={k:int(np.count_nonzero(z['seeds'][i]>=0)) for i,k in enumerate(meta['channels'])}
    item['baseline_normalization_relative_stat']=float(np.sqrt(max(c[0,0],0))/abs(m[0]))
    output[ch]=item
for tag in (['eta20','eta27','eta30'] if args.charge=='both' else []):
    lo,hi=meta['slices'][tag];i=np.arange(lo,hi)
    p,n=means['WP'][i],means['WM'][i]
    v=(2*n/(p+n)**2)**2*np.diag(covs['WP'])[i]+(2*p/(p+n)**2)**2*np.diag(covs['WM'])[i]
    pb,nb=boots['WP'][:,i],boots['WM'][:,i]
    output['asymmetry_'+tag+'_absolute_stat']=((pb-nb)/(pb+nb)).std(axis=0,ddof=1).tolist()
    output['asymmetry_'+tag+'_target_met']=bool(max(output['asymmetry_'+tag+'_absolute_stat'])<=.002)
    passed.append(output['asymmetry_'+tag+'_target_met'])
output['all_available_precision_targets_met']=all(passed)
output['all_three_precision_criteria_checked']=args.charge=='both'
print(json.dumps(output,indent=2))
