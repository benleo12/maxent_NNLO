"""Exclusive physical-task manifest, frozen before production submission."""
from pathlib import Path
import json,argparse,hashlib,time,re
ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--training-replicas',type=int,default=160);ap.add_argument('--validation-replicas',type=int,default=64);ap.add_argument('--groups',default='TV');ap.add_argument('--output',default='production_initial');a=ap.parse_args()
old=set(json.loads((a.root/'scripts/previous_physical_seeds.json').read_text()))
tasks=[];seen=set()
for ci,ch in enumerate(['WP','WM']):
 for ji,channel in enumerate(['LO','R','V','RR','RV','VV']):
  for group in ['T','V']:
   if group not in a.groups:continue
   card=a.root/'cards'/ch/group/channel/'rc.run';digest=hashlib.sha256(card.read_bytes()).hexdigest()
   budget=re.search(r'production\s*=\s*(\d+)\[(\d+)\]',card.read_text());points=int(budget.group(1))*int(budget.group(2))
   for i in range(a.training_replicas if group=='T' else a.validation_replicas):
    seed=1000+ci*10000+ji*1000+i if group=='T' else 30000+ci*30000+ji*4000+i
    assert 0<seed<100000 and seed not in old and seed not in seen
    seen.add(seed)
    tasks.append(dict(charge=ch,channel=channel,group=group,replica=i,seed=seed,points_per_run=points,card_sha256=digest))
priority={c:i for i,c in enumerate(['RR','RV','VV','V','R','LO'])}
tasks.sort(key=lambda t:(priority[t['channel']],t['charge'],t['group'],t['replica']))
record=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol_sha256=hashlib.sha256((a.root/'scripts/PROTOCOL.md').read_bytes()).hexdigest(),
 previous_physical_seed_count=len(old),seed_overlap_with_previous=0,partition='ordered tasks sliced by shard index, exactly one node per shard',tasks=tasks)
dest=a.root/a.output;dest.mkdir(exist_ok=True)
out=dest/'manifest.json';assert not out.exists();out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(tasks=len(tasks),integration_points=sum(r['points_per_run'] for r in tasks),seeds_disjoint=True,manifest=str(out))))
