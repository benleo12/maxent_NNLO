"""Prepare a compact, auditable manuscript handoff after validation is complete."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
out={}
for ch in ['WP','WM']:
 f=json.loads((ROOT/'output'/ch/'MODEL_FREEZE.json').read_text())
 v=json.loads((ROOT/'output'/ch/'validation.json').read_text())
 t=np.load(str(ROOT/'input'/('T_%s.npz'%ch)))
 sel=f['selected'];central=f['models'][sel]['scales'][0];sig=t['mean'][0]
 models={}
 for m in ['P','B','A','J2','J4']:
  r=v['models'][m]
  models[m]=dict(acceptance=r['acceptance'],rms_relative_distribution_bias={k:x['relative_rms'] for k,x in r['distributions'].items()})
 out[ch]=dict(selected=sel,orders=f['orders'],angular_order=6,n_fid=f['prior']['n_fid'],
  prior_rate_pb=f['prior']['prior_sigma_fid_pb'],training_rate_pb=float(sig[0]),
  training_rate_MC_pb=float(np.sqrt(t['cov'][0,0])),training_rate_scale_down_pb=float(sig[0]-sig.min()),
  training_rate_scale_up_pb=float(sig.max()-sig[0]),selected_diagnostics=central,
  max_target_pull_all_scales=max(r['max_pull'] for r in f['models'][sel]['scales']),
  training_block_tests={m:f['models'][m]['training_test'] for m in ['B','A','J2']},
  selected_linear_response_checks=v['linear_response_checks'],models=models,
  training_freeze_utc=f['freeze_utc'],training_input_sha=f['input_sha256'],validation_input_sha=v['validation_input_sha256'])
asym=json.loads((ROOT/'output/charge_asymmetry.json').read_text())
out['asymmetry']={m:{tag:{k:r[k] for k in ['rms_absolute_bias','statistical_comparison']} for tag,r in rows.items()} for m,rows in asym.items() if m!='reference'}
(ROOT/'output/RESULTS_SUMMARY.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(ROOT/'output/RESULTS_SUMMARY.json')
