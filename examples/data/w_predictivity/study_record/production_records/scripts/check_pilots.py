"""Compare identical-seed integration inputs without opening new predictions."""
from pathlib import Path
import json,argparse,numpy as np
ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--pilot-tag',default='pilot');a=ap.parse_args();root=a.root
checks=[]
for record in sorted((root/a.pilot_tag).glob('*/*/*/s*/complete.json')):
 d=json.loads(record.read_text());assert d['ok']
 group=d['group'];charge=d['charge'];channel=d['channel'];seed=d['seed']
 old=root/'baseline_pilot'/group/charge/channel/('s%d'%seed)
 assert json.loads((old/'complete.json').read_text())['ok']
 oldprefix='%s.W_VALID_%s.%s.'%(charge,group,channel)
 newprefix='%s.W_PRED_%s.%s.'%(charge,group,channel)
 tags=['norm_born']
 if group=='T':
  tags=[p.name[len(oldprefix):].split('.s')[0] for p in old.glob(oldprefix+'*.s%d.dat'%seed)]
 maximum=0.
 for tag in tags:
  p=np.loadtxt(old/(oldprefix+tag+'.s%d.dat'%seed),ndmin=2)
  q=np.loadtxt(record.parent/(newprefix+tag+'.s%d.dat'%seed),ndmin=2)
  assert p.shape==q.shape
  scale=max(np.max(np.abs(p[:,3:])),1e-20)
  delta=float(np.max(np.abs(p[:,3:]-q[:,3:]))/scale)
  assert delta<2e-10,(charge,channel,group,tag,delta)
  maximum=max(maximum,delta)
 if group=='T' and a.pilot_tag!='pilot':
  previous=root/'pilot'/group/charge/channel/('s%d'%seed)
  for path in record.parent.glob(newprefix+'pred_*.s%d.dat'%seed):
   p=np.loadtxt(previous/path.name,ndmin=2);q=np.loadtxt(path,ndmin=2)
   assert np.allclose(p,q,rtol=2e-10,atol=1e-12),path
 if group=='V' and a.pilot_tag in ['pilot_final','pilot_fastpath','pilot_ready']:
  previous=root/'pilot_cached'/group/charge/channel/('s%d'%seed)
  regions=[('qc','pt0'),('qc','pt27'),('qc','pt30'),('qc','proj27'),('qc','proj30'),('qm','pt0'),('qm','pt27'),('qm','pt30')]
  for k,(category,selection) in enumerate(regions,1):
   packed=np.loadtxt(record.parent/(newprefix+'region%d.s%d.dat'%(k,seed)),ndmin=2)
   for iq in range(3):
    for ir in range(3):
     tag='expl_%s_%d_%d_%s'%(category,iq,ir,selection)
     single=np.loadtxt(previous/(newprefix+tag+'.s%d.dat'%seed),ndmin=2)
     p=single[0,3:]*(single[0,2]-single[0,0]);q=packed[3*iq+ir,3:]*(packed[3*iq+ir,2]-packed[3*iq+ir,0])
     assert np.allclose(p,q,rtol=2e-10,atol=1e-8),(charge,channel,tag,float(abs(p-q).max()))
 if group=='T' and channel=='LO':
  norm=np.loadtxt(record.parent/(newprefix+'norm_born.s%d.dat'%seed),ndmin=2)[0,3::2]
  for path in record.parent.glob(newprefix+'pred_[HS]_*.s%d.dat'%seed):
   x=np.loadtxt(path,ndmin=2)[0,3::2]
   assert np.max(abs(x)/np.maximum(abs(norm),1e-30))<1e-10,path
  for iq in range(3) if a.pilot_tag=='pilot' else []:
   for ia in range(3):
    for cut in [27,30]:
     name='expl_qc_%d_%d_'%(iq,ia)
     p=np.loadtxt(record.parent/(newprefix+name+'pt%d.s%d.dat'%(cut,seed)),ndmin=2)
     q=np.loadtxt(record.parent/(newprefix+name+'proj%d.s%d.dat'%(cut,seed)),ndmin=2)
     assert np.allclose(p,q,rtol=2e-10,atol=1e-12),(iq,ia,cut)
 checks.append(dict(charge=charge,channel=channel,group=group,common_tables=len(tags),max_relative_difference=maximum))
report=dict(checks=checks,completed=24,passed=True,validation_blinding='Only baseline normalization inspected in validation pilots; new selected rates/spectra not read.')
assert len(checks)==24
(root/'technical'/('INTEGRATION_%s_CHECK.json'%a.pilot_tag.upper())).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(completed=len(checks),passed=True,max_relative_difference=max(d['max_relative_difference'] for d in checks))))
