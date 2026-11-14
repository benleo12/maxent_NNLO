"""Pool independent NNLOJET runs, preserving channel and scale correlations.

No channel is weighted by its estimated variance: every equal-budget seed is
averaged, and independently integrated channels are then added.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

CHANNELS = ['LO','R','V','RR','RV','VV']

def training_tags():
    return (['norm_born'] + ['prof_mw_%d'%n for n in range(1,13)]
            + ['prof_absyw_%d'%n for n in range(1,13)]
            + ['prof_wptw_%d'%n for n in range(21)]
            + ['prof_wctw_%d'%n for n in range(7)]
            + ['prof_yc_%d%d'%(k,n) for k in range(1,5) for n in range(5)])

def aggregate(root, group, charge, expected, rr_replicas=None, extra_root=None):
    tags = training_tags() if group=='T' else ['norm_born','eta20','eta27','eta30','ptl','mt','recoil']
    channel_acc, channel_err, all_seeds, records = [], [], [], []
    labels = None
    slices, edges = {}, {}
    for channel in CHANNELS:
        runs=sorted((root/'runs'/group/charge/channel).glob('s*/complete.json'))
        count = rr_replicas if group=='V' and channel=='RR' and rr_replicas else expected
        if extra_root is not None and group=='V' and channel=='RR':
            runs+=sorted((extra_root/'runs'/group/charge/channel).glob('s*/complete.json'))
            runs=sorted(runs,key=lambda p:int(p.parent.name[1:]))
        if len(runs)!=count:
            raise ValueError('%s/%s/%s: %d completed, expected %d'%(group,charge,channel,len(runs),count))
        acc,err,seeds=[],[],[]
        for record in runs:
            rec=json.loads(record.read_text())
            assert rec['ok']
            assert rec['seed'] not in seeds, 'Duplicated physical seed'
            values,errors,lab=[],[],[]
            offset=0
            for tag in tags:
                fn=record.parent/('%s.W_VALID_%s.%s.%s.s%d.dat'%(charge,group,channel,tag,rec['seed']))
                a=np.loadtxt(str(fn),ndmin=2)
                assert a.shape[1]==17, (fn,a.shape)
                width=a[:,2]-a[:,0]
                values.append(a[:,3::2]*width[:,None]/1000.) # integrated pb
                errors.append(a[:,4::2]*width[:,None]/1000.)
                lab += [tag] if len(a)==1 else ['%s_%02d'%(tag,j) for j in range(len(a))]
                slices[tag]=[offset,offset+len(a)]
                edges[tag]=np.r_[a[:,0],a[-1,2]].tolist()
                offset+=len(a)
            if labels is None: labels=lab
            assert lab==labels
            acc.append(np.concatenate(values));err.append(np.concatenate(errors));seeds.append(rec['seed'])
            records.append(rec)
        channel_acc.append(acc);channel_err.append(err);all_seeds.append(seeds)
    channel_acc=[np.asarray(a) for a in channel_acc]
    counts=np.array([len(a) for a in channel_acc])
    cov=sum(np.cov(a[:,:,0],rowvar=False,ddof=1)/len(a) for a in channel_acc)
    if len(set(counts))==1:
        # Keep byte-for-byte numerical equivalence with the original uniform run set.
        acc=np.asarray(channel_acc);errors=np.asarray(channel_err);seeds=np.asarray(all_seeds)
        mean=acc.mean(axis=1).sum(axis=0)
    else:
        mean=np.asarray([a.mean(axis=0) for a in channel_acc]).sum(axis=0)
        acc=np.full((len(CHANNELS),max(counts))+channel_acc[0].shape[1:],np.nan)
        errors=np.full_like(acc,np.nan);seeds=np.full((len(CHANNELS),max(counts)),-1,dtype=int)
        for i,n in enumerate(counts):
            acc[i,:n]=channel_acc[i];errors[i,:n]=channel_err[i];seeds[i,:n]=all_seeds[i]
    meta=dict(group=group,charge=charge,channels=CHANNELS,replicas=expected,
              points_per_run=250000,labels=labels,slices=slices,edges=edges,
              units='pb integrated in each bin; profiles are signed weighted sums',
              estimator='equal-weight mean per channel, followed by sum of independent channels',
              covariance='sum over channels of sample covariance across seeds divided by seed count',
              replicas_by_channel=dict(zip(CHANNELS,counts.tolist())),
              padding='Unused rows are NaN with seed -1; n_runs gives the physical rows in each channel.',
              records=records)
    return dict(acc=acc,integration_errors=errors,seeds=seeds,n_runs=counts,
                mean=mean,cov=cov,metadata=np.array(json.dumps(meta)))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--group',choices=['T','V'],required=True)
    ap.add_argument('--replicas',type=int,default=32)
    ap.add_argument('--rr-replicas',type=int)
    ap.add_argument('--extra-root',type=Path)
    ap.add_argument('--charge',choices=['WP','WM','both'],default='both')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for charge in (['WP','WM'] if args.charge=='both' else [args.charge]):
        result=aggregate(args.root,args.group,charge,args.replicas,args.rr_replicas,args.extra_root)
        out=args.output/('%s_%s.npz'%(args.group,charge))
        if out.exists():
            old=np.load(str(out),allow_pickle=False)
            assert all(np.array_equal(old[k],result[k],equal_nan=True) for k in ['acc','seeds','mean','cov']), 'Existing input changed; preserve it and use a new version'
        else:
            np.savez_compressed(str(out),**result)
        print(json.dumps(dict(file=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                              shape=list(result['acc'].shape))))
