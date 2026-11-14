from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time

ap=argparse.ArgumentParser()
ap.add_argument('--pilot',action='store_true')
ap.add_argument('--workers',type=int,default=64)
ap.add_argument('--shard',type=int,default=0)
ap.add_argument('--shards',type=int,default=1)
ap.add_argument('--replicas',type=int,default=32)
ap.add_argument('--iteration-points',type=int,default=50000)
ap.add_argument('--iterations',type=int,default=5)
ap.add_argument('--manifest',type=Path)
args=ap.parse_args()
root=Path(os.environ.get('WVAL_ROOT','/pscratch/sd/b/bassi/jhep_w_validation_20261113'))
binary=root/'install/bin/NNLOJET'
assert binary.exists()
env=dict(os.environ)
env['LD_LIBRARY_PATH']=str(root/'install/lib')+':/global/cfs/cdirs/m4599/Projects/sherpa/bassi/software/LHAPDF-6.5.5/local/lib:'+env.get('LD_LIBRARY_PATH','')
env['LHAPDF_DATA_PATH']='/global/cfs/cdirs/m4599/Projects/sherpa/shoeche/lhapdf/share/LHAPDF'
if str(root).startswith('/home/'):
    env['LD_LIBRARY_PATH']=str(root/'install/lib')+':/home/bassi/w_minnlo/lhapdf/lib:'+env.get('LD_LIBRARY_PATH','')
    env['LHAPDF_DATA_PATH']='/home/bassi/w_minnlo/lhapdf/share/LHAPDF'
env['OMP_NUM_THREADS']='1'
channels=['LO','R','V','RR','RV','VV']
tasks=[]
for charge_i,chg in enumerate(['WP','WM']):
    for channel_i,channel in enumerate(channels):
        for group_i,group in enumerate(['T','V']):
            count=1 if args.pilot else args.replicas
            for i in range(count):
                seed=(510000+group_i*100000 if args.pilot else 710000+group_i*200000)+charge_i*20000+channel_i*1000+i
                tasks.append((chg,channel,group,i,seed))
if args.manifest:
    claimed=json.loads(args.manifest.read_text())
    tasks=[tuple(rec[k] for k in ['charge','channel','group','replica','seed']) for rec in claimed['tasks']]
tasks=tasks[args.shard::args.shards]

def run(task):
    chg,channel,group,i,seed=task
    carddir=root/'cards'/chg/group/channel
    parent='pilot' if args.pilot else 'runs'
    d=root/parent/group/chg/channel/('s%d'%seed)
    d.mkdir(parents=True,exist_ok=True)
    complete=d/'complete.json'
    if complete.exists():
        return json.loads(complete.read_text())
    card=(carddir/('pilot.run' if args.pilot else 'rc.run')).read_text()
    if not args.pilot:
        import re
        card=re.sub(r'production\s*=.*','production = %d[%d]'%(args.iteration_points,args.iterations),card)
    (d/'rc.run').write_text(card)
    gridname='%s.W_VALID_%s.y1.00E-07.%s'%(chg,group,channel)
    shutil.copy2(carddir/gridname,d/gridname)
    start=time.time()
    with (d/'run.log').open('w') as log:
        result=subprocess.run([str(binary),'--run','rc.run','--iseed',str(seed)],cwd=str(d),env=env,stdout=log,stderr=subprocess.STDOUT)
    output=(d/'run.log').read_text(errors='replace')
    ok=result.returncode==0 and 'Elapsed time' in output
    rec=dict(charge=chg,channel=channel,group=group,replica=i,seed=seed,
             elapsed_s=time.time()-start,returncode=result.returncode,ok=ok,directory=str(d),
             card_sha256=hashlib.sha256(card.encode()).hexdigest())
    (complete if ok else d/'failed.json').write_text(json.dumps(rec,indent=2)+'\n')
    return rec

start=time.time()
bad=[]
print('Starting %d independent runs, workers=%d, shard=%d/%d'%(len(tasks),args.workers,args.shard,args.shards),flush=True)
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures=[pool.submit(run,task) for task in tasks]
    for future in as_completed(futures):
        rec=future.result()
        print(json.dumps(rec),flush=True)
        if not rec['ok']:
            bad.append(rec)
print('Finished in %.1fs; failures=%d'%(time.time()-start,len(bad)),flush=True)
if bad:
    raise SystemExit(1)
