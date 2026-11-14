"""Audit final physical seed counts, independence and frozen-model provenance."""
from pathlib import Path
import hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected={'WP':[864,2976,96,3296,768,32],'WM':[1056,1760,64,1824,736,32]}
all_seeds=set();inputs={};total=0
for group in ['T','V']:
 for charge in ['WP','WM']:
  p=ROOT/'input'/('%s_%s.npz'%(group,charge));z=np.load(str(p),allow_pickle=False);meta=json.loads(str(z['metadata']))
  counts=np.sum(z['seeds']>=0,axis=1).tolist();assert counts==([32]*6 if group=='T' else expected[charge])
  valid=z['seeds']>=0;seeds=set(z['seeds'][valid].tolist());assert len(seeds)==sum(counts)
  assert not (seeds & all_seeds), 'Integration seeds overlap between groups, charges or channels'
  all_seeds.update(seeds);total+=len(seeds)
  assert np.isfinite(z['acc'][valid]).all();assert np.isnan(z['acc'][~valid]).all()
  assert np.isfinite(z['mean']).all() and np.isfinite(z['cov']).all()
  inputs[group+'_'+charge]=dict(sha256=sha(p),channels=meta['channels'],counts=counts,
      points_per_run=250000,physical_runs=len(seeds),accumulator_shape=list(z['acc'].shape),
      seed_list_sha256=hashlib.sha256(z['seeds'][valid].tobytes()).hexdigest(),
      seed_range=[min(seeds),max(seeds)],covariance_estimator=meta['covariance'])
freezes={ch:json.loads((ROOT/'output'/ch/'MODEL_FREEZE.json').read_text()) for ch in ['WP','WM']}
complete=json.loads((ROOT/'output/VALIDATION_COMPLETE.json').read_text())
manifest=dict(schema='w-prediction-study-final-manifest-v1',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 inputs=inputs,total_physical_runs=total,total_integration_points=total*250000,
 seed_groups_disjoint=True,all_physical_accumulators_finite=True,
 selected={ch:f['selected'] for ch,f in freezes.items()},freeze_utc={ch:f['freeze_utc'] for ch,f in freezes.items()},
 weight_sha256={ch:{m:r['weights_sha256'] for m,r in f['models'].items()} for ch,f in freezes.items()},
 validation_complete=complete,actual_precision_audit_sha256=sha(ROOT/'input/precision_only.json'),
 training_package_snapshot='b437a686f1282b0567010518c7b3b2daae838ab7',
 propagation='Independent channel-wise whole-vector integration resampling; prediction input response is linear. Finite-prior influence functions include the fitted response, blended recoil targets and prior-standardized ridge. Model differences retain cross-model covariance.',
 limitations='Scale alternatives are coherent seven-point vectors, not Gaussian errors or coverage intervals. No PDF-member integrations are added by this study. A finite moment fit establishes only the tested predictions, not universal NNLO+PS accuracy.',
 final_precision_decision='After the last precision-only allocation, unblind once and report actual precision including any remaining misses; no model retuning or seed deletion.')
(ROOT/'output/FINAL_MANIFEST.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(total_physical_runs=total,selected=manifest['selected'],seed_groups_disjoint=True)))
