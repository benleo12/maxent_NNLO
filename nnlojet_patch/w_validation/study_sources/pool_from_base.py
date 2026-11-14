"""Pool physical extensions directly, preserving the original compact seed rows."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,numpy as np
ROOT=Path('/pscratch/sd/b/bassi/jhep_w_validation_20261113')
CHANNELS=['LO','R','V','RR','RV','VV']
COUNTS={'WP':[864,2976,96,3296,768,32],'WM':[1056,1760,64,1824,736,32]}
COLLECTIONS=[('rr_extension',61256667),('precision_extension2',61257244),('precision_extension3',61258856)]
TAGS=['norm_born','eta20','eta27','eta30','ptl','mt','recoil']
def channel_data(task):
 ch,c=task;ci=CHANNELS.index(c);base=np.load(str(ROOT/'initial_compact'/('V_%s.npz'%ch)),allow_pickle=False);meta=json.loads(str(base['metadata']))
 acc=list(base['acc'][ci]);err=list(base['integration_errors'][ci]);seeds=base['seeds'][ci].tolist()
 records=[r for r in meta['records'] if r['channel']==c]
 for collection,job in COLLECTIONS:
  manifest=json.loads((ROOT/collection/'manifest.json').read_text())
  for t in manifest['tasks']:
   if (t['charge'],t['channel'])!=(ch,c):continue
   d=ROOT/'runs/V'/ch/c/('s%d'%t['seed']);rec=json.loads((d/'complete.json').read_text())
   assert rec['ok'] and rec['returncode']==0 and rec['seed']==t['seed']
   assert rec['seed'] not in seeds
   values=[];errors=[]
   for tag in TAGS:
    fn=d/('%s.W_VALID_V.%s.%s.s%d.dat'%(ch,c,tag,t['seed']))
    a=np.loadtxt(str(fn),ndmin=2);assert a.shape[1]==17
    assert np.allclose(np.r_[a[:,0],a[-1,2]],meta['edges'][tag],rtol=0,atol=1e-12)
    width=a[:,2]-a[:,0];values.append(a[:,3::2]*width[:,None]/1000.);errors.append(a[:,4::2]*width[:,None]/1000.)
   v=np.concatenate(values);e=np.concatenate(errors)
   assert np.isfinite(v).all() and np.isfinite(e).all()
   acc.append(v);err.append(e);seeds.append(rec['seed'])
   rec.update(origin_cluster='Perlmutter',origin_job=job,origin_directory=rec['directory'],collection=collection)
   records.append(rec)
 assert len(seeds)==COUNTS[ch][ci]
 ix=np.argsort(seeds)
 return ch,c,np.asarray(acc)[ix],np.asarray(err)[ix],np.asarray(seeds)[ix],[records[i] for i in ix]
if __name__=='__main__':
 out=ROOT/'compact_fast';out.mkdir(exist_ok=True)
 with ProcessPoolExecutor(max_workers=8) as pool:rows=list(pool.map(channel_data,[(ch,c) for ch in ['WP','WM'] for c in CHANNELS]))
 for ch in ['WP','WM']:
  basepath=ROOT/'initial_compact'/('V_%s.npz'%ch);base=np.load(str(basepath),allow_pickle=False);meta=json.loads(str(base['metadata']))
  parts=[r for r in rows if r[0]==ch];assert [r[1] for r in parts]==CHANNELS
  counts=np.array(COUNTS[ch]);acc=np.full((6,max(counts),67,7),np.nan);err=np.full_like(acc,np.nan);seeds=np.full((6,max(counts)),-1,dtype=int)
  records=[]
  for i,r in enumerate(parts):
   acc[i,:counts[i]]=r[2];err[i,:counts[i]]=r[3];seeds[i,:counts[i]]=r[4];records+=r[5]
  mean=np.asarray([r[2].mean(axis=0) for r in parts]).sum(axis=0)
  cov=sum(np.cov(r[2][:,:,0],rowvar=False,ddof=1)/len(r[2]) for r in parts)
  meta.update(replicas_by_channel=dict(zip(CHANNELS,counts.tolist())),records=records,
   padding='Unused rows are NaN with seed -1; n_runs gives actual physical rows.',
   original_compact_input_sha256=hashlib.sha256(basepath.read_bytes()).hexdigest(),
   pooling='Original 32 physical rows/channel plus every successful extension seed; equal means per channel, summed channels. Raw archive transfer is separate.')
  fn=out/('V_%s.npz'%ch)
  np.savez_compressed(str(fn),acc=acc,integration_errors=err,seeds=seeds,n_runs=counts,mean=mean,cov=cov,metadata=np.array(json.dumps(meta)))
  print(json.dumps(dict(file=str(fn),sha256=hashlib.sha256(fn.read_bytes()).hexdigest(),counts=counts.tolist())),flush=True)
