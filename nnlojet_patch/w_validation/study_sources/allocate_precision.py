"""Allocate independent runs using only asymmetry-variance contributions."""
from pathlib import Path
import json,numpy as np,statistics
root=Path('/home/bassi/jhep_w_validation_20261113/compact_final')
z={ch:np.load(str(root/('V_%s.npz'%ch)),allow_pickle=False) for ch in ['WP','WM']}
meta={ch:json.loads(str(z[ch]['metadata'])) for ch in z}
inds=np.concatenate([np.arange(*meta['WP']['slices'][tag]) for tag in ['eta20','eta27','eta30']])
p=z['WP']['mean'][inds,0];m=z['WM']['mean'][inds,0]
j={'WP':2*m/(p+m)**2,'WM':-2*p/(p+m)**2}
coeff=[];counts=[];cost=[];keys=[]
for ch in ['WP','WM']:
 for i,c in enumerate(meta[ch]['channels']):
  rows=z[ch]['acc'][i,:,:,0];rows=rows[np.isfinite(rows).all(1)];n=len(rows)
  coeff.append(np.var(rows[:,inds],axis=0,ddof=1)*j[ch]**2);counts.append(n);keys.append((ch,c))
  records=[r for r in meta[ch]['records'] if r['channel']==c]
  per=[r['elapsed_s'] for r in records if r.get('origin_cluster')=='Perlmutter']
  cost.append(statistics.median(per or [r['elapsed_s'] for r in records]))
a=np.array(coeff).T;n0=np.array(counts);cost=np.array(cost)
out=[]
for target in [.00195,.0019,.0018]:
 n=n0.copy()
 for iteration in range(10000):
  v=(a/n).sum(1)
  if v.max()<=target**2:break
  active=v>target**2;weight=np.maximum(v-target**2,0)
  gain=a*(1/n-1/(n+32))
  score=(weight[:,None]*gain).sum(0)/(32*cost)
  k=int(score.argmax());n[k]+=32
 total=float(np.sum((n-n0)*cost));new=int((n-n0).sum())
 out.append(dict(target_max_absolute_error=target,predicted_max_absolute_error=float(np.sqrt(v.max())),additional_runs=new,
  counts={ch:{c:int(n[k]) for k,(cc,c) in enumerate(keys) if cc==ch} for ch in ['WP','WM']},
  cpu_hours=total/3600,ideal_wall_minutes_4nodes=total/(512*60),ideal_wall_minutes_8nodes=total/(1024*60),
  costs_seconds={ch:{c:float(cost[k]) for k,(cc,c) in enumerate(keys) if cc==ch} for ch in ['WP','WM']}))
print(json.dumps(out,indent=2))
