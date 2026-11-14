"""Diagnose precision sources without printing held-out central predictions."""
from pathlib import Path
import argparse,json
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);a=ap.parse_args()
out={}
for ch in ['WP','WM']:
 z=np.load(str(a.input/('V_%s.npz'%ch)),allow_pickle=False);m=z['mean'][:,0];meta=json.loads(str(z['metadata']));out[ch]={}
 for ci,channel in enumerate(meta['channels']):
  raw=z['acc'][ci,:,:,0];raw=raw[np.isfinite(raw).all(1)];delta=raw-raw.mean(0);n=len(raw);c=delta.T@delta/(n*(n-1))
  row=dict(n_runs=n,normalization_relative_error=float(np.sqrt(c[0,0])/abs(m[0])),cuts={})
  for cut in [20,27,30]:
   lo,hi=meta['slices']['eta%d'%cut];ix=np.arange(lo,hi);accept=m[ix].sum()/m[0]
   j=np.zeros(len(m));j[ix]=1/m[0];j[0]-=accept/m[0];d=delta@j;v=np.sum(d*d)
   row['cuts'][str(cut)]=dict(acceptance_relative_error=float(np.sqrt(v/(n*(n-1)))/abs(accept)),
    largest_acceptance_seed_variance_fraction=float(np.max(d*d)/max(v,1e-300)),
    bin_relative_errors=(np.sqrt(np.maximum(np.diag(c)[ix],0))/np.abs(m[ix])).tolist())
  out[ch][channel]=row
print(json.dumps(out,indent=2,allow_nan=False))
