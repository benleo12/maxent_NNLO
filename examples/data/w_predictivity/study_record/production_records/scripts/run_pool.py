from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import threading

ap=argparse.ArgumentParser()
ap.add_argument('--pilot',action='store_true')
ap.add_argument('--baseline',action='store_true')
ap.add_argument('--pilot-points',type=int,default=1000)
ap.add_argument('--pilot-iterations',type=int,default=2)
ap.add_argument('--pilot-tag',default='pilot')
ap.add_argument('--install-subdir',default='install')
ap.add_argument('--pin-workers',action='store_true')
ap.add_argument('--groups')
ap.add_argument('--workers',type=int,default=64)
ap.add_argument('--shard',type=int,default=0)
ap.add_argument('--shards',type=int,default=1)
ap.add_argument('--replicas',type=int,default=64)
ap.add_argument('--iteration-points',type=int)
ap.add_argument('--iterations',type=int)
ap.add_argument('--manifest',type=Path)
args=ap.parse_args()
root=Path(os.environ.get('WPRED_ROOT','/pscratch/sd/b/bassi/jhep_w_predictivity_20261114'))
source=root if not args.baseline else root.parent/'jhep_w_validation_20261113'
assert not args.baseline or args.pilot,'Baseline reruns are technical pilots only'
install=source/('install' if args.baseline else args.install_subdir)
binary=install/'bin/NNLOJET'
assert binary.exists()
env=dict(os.environ)
env['LD_LIBRARY_PATH']=str(install/'lib')+':/global/cfs/cdirs/m4599/Projects/sherpa/bassi/software/LHAPDF-6.5.5/local/lib:'+env.get('LD_LIBRARY_PATH','')
env['LHAPDF_DATA_PATH']='/global/cfs/cdirs/m4599/Projects/sherpa/shoeche/lhapdf/share/LHAPDF'
if str(root).startswith('/home/'):
    env['LD_LIBRARY_PATH']=str(root/'install/lib')+':/home/bassi/w_minnlo/lhapdf/lib:'+env.get('LD_LIBRARY_PATH','')
    env['LHAPDF_DATA_PATH']='/home/bassi/w_minnlo/lhapdf/share/LHAPDF'
env['OMP_NUM_THREADS']='1'
channels=['LO','R','V','RR','RV','VV']
groups=set(args.groups or 'TV')
pilot_plan=root/'pilot_plans'/(args.pilot_tag+'.json')
if args.pilot and args.groups is None and pilot_plan.exists():
    groups=set(json.loads(pilot_plan.read_text())['groups'])
assert groups and groups<=set('TV')
tasks=[]
expected_cards={}
for charge_i,chg in enumerate(['WP','WM']):
    for channel_i,channel in enumerate(channels):
        for group_i,group in enumerate(['T','V']):
            if group not in groups:continue
            count=1 if args.pilot else args.replicas
            for i in range(count):
                if args.pilot: seed=95000+charge_i*1000+channel_i*100+group_i*10
                elif group=='T': seed=1000+charge_i*10000+channel_i*1000+i
                else: seed=30000+charge_i*30000+channel_i*4000+i
                assert 0<seed<1000000
                tasks.append((chg,channel,group,i,seed))
if args.manifest:
    claimed=json.loads(args.manifest.read_text())
    tasks=[tuple(rec[k] for k in ['charge','channel','group','replica','seed']) for rec in claimed['tasks']]
    expected_cards={rec['seed']:rec['card_sha256'] for rec in claimed['tasks']}
assert len({t[4] for t in tasks})==len(tasks),'Seeds must be disjoint in a campaign'
tasks=tasks[args.shard::args.shards]
worker_state=threading.local();pin_lock=threading.Lock();physical=[]
if args.pin_workers:
    seen_cores=set()
    for cpu in sorted(os.sched_getaffinity(0)):
        top=Path('/sys/devices/system/cpu/cpu%d/topology'%cpu)
        key=((top/'physical_package_id').read_text().strip(),(top/'core_id').read_text().strip())
        if key not in seen_cores:physical.append(cpu);seen_cores.add(key)
    assert args.workers<=len(physical),'Requested workers exceed assigned physical cores'
    print('Assigned %d logical CPUs, %d physical cores; pinning %d workers'%(len(os.sched_getaffinity(0)),len(physical),args.workers),flush=True)
core_iterator=iter(physical)

def run(task):
    if args.pin_workers and not hasattr(worker_state,'cpu'):
        with pin_lock:worker_state.cpu=next(core_iterator)
        os.sched_setaffinity(0,{worker_state.cpu})
    chg,channel,group,i,seed=task
    carddir=source/'cards'/chg/group/channel
    parent=args.pilot_tag if args.pilot else 'runs'
    if args.baseline: parent='baseline_pilot'
    d=root/parent/group/chg/channel/('s%d'%seed)
    d.mkdir(parents=True,exist_ok=True)
    complete=d/'complete.json'
    if complete.exists():
        prior=json.loads(complete.read_text())
        assert prior['ok'] and prior['seed']==seed
        if seed in expected_cards:assert prior['card_sha256']==expected_cards[seed]
        return prior
    claim=d/'claimed'
    claim.mkdir()  # Atomic exclusive claim; an overlapping worker must stop.
    (claim/'owner.json').write_text(json.dumps(dict(pid=os.getpid(),host=os.uname().nodename,
        job=os.environ.get('SLURM_JOB_ID'),seed=seed,time=time.time()),indent=2)+'\n')
    card=(carddir/'rc.run').read_text()
    import re
    if args.pilot: card=re.sub(r'production\s*=.*','production = %d[%d]'%(args.pilot_points,args.pilot_iterations),card)
    if not args.pilot and args.iteration_points is not None:
        import re
        card=re.sub(r'production\s*=.*','production = %d[%d]'%(args.iteration_points,args.iterations or 5),card)
    if seed in expected_cards:
        assert hashlib.sha256(card.encode()).hexdigest()==expected_cards[seed],'Card differs from frozen manifest'
    budget=re.search(r'production\s*=\s*(\d+)\[(\d+)\]',card)
    points_per_run=int(budget.group(1))*int(budget.group(2))
    (d/'rc.run').write_text(card)
    gridname='%s.%s_%s.y1.00E-07.%s'%(chg,'W_VALID' if args.baseline else 'W_PRED',group,channel)
    shutil.copy2(carddir/gridname,d/gridname)
    start=time.time()
    with (d/'run.log').open('w') as log:
        result=subprocess.run([str(binary),'--run','rc.run','--iseed',str(seed)],cwd=str(d),env=env,stdout=log,stderr=subprocess.STDOUT)
    output=(d/'run.log').read_text(errors='replace')
    ok=result.returncode==0 and 'Elapsed time' in output
    rec=dict(charge=chg,channel=channel,group=group,replica=i,seed=seed,
             elapsed_s=time.time()-start,returncode=result.returncode,ok=ok,directory=str(d),
             card_sha256=hashlib.sha256(card.encode()).hexdigest(),variant='baseline' if args.baseline else 'new',
             points_per_run=points_per_run,
             cpu_affinity=sorted(os.sched_getaffinity(0)),install_prefix=str(install),
             binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
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
