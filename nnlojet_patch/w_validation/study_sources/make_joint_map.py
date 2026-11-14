"""Cache the W- J2/A redistribution on exactly the same accepted prior events."""
from pathlib import Path
import json,numpy as np
from train import ROOT,load_prior,sha
ch='WM';ev,p,pm=load_prior(ch);d=ROOT/'output'/ch
f=json.loads((d/'MODEL_FREEZE.json').read_text());assert f['selected']=='J2'
a=np.load(str(d/'A_frozen.npz'));j=np.load(str(d/'J2_frozen.npz'))
assert sha(d/'A_frozen.npz')==f['models']['A']['weights_sha256']
assert sha(d/'J2_frozen.npz')==f['models']['J2']['weights_sha256']
qa=a['weights'][0];qj=j['weights'][0];ye=np.linspace(0,4,17);ce=np.linspace(-1,1,21)
def hist(w):return np.histogram2d(ev['y_abs'],ev['ctw'],bins=(ye,ce),weights=w)[0]
n=hist(None);ha=hist(qa);hj=hist(qj);sq=hist(qa*qa)
eff=np.divide(ha*ha,sq,out=np.zeros_like(ha),where=sq>0)
ratio=np.divide(hj,ha,out=np.full_like(ha,np.nan),where=ha>0)
valid=(n>=100)&(eff>=100)&np.isfinite(ratio)
meta=dict(process='W-',comparison='J2/A on the identical prior events; redistribution map, not NNLO closure',
 baseline_cuts=dict(pt_l_min_GeV=20.,pt_nu_min_GeV=25.,mt_min_GeV=40.,abs_eta_l_max=2.5),fiducial_event_count=len(p),
 fiducial_index_sha256=f['prior']['fiducial_index_sha256'],
 angle='signed charged-lepton Collins-Soper cosine with axis folded along yW',
 mask='raw count >=100 and A-weighted effective count >=100 per cell',
 probability_in_map_A=float(ha.sum()),probability_in_map_J2=float(hj.sum()),
 A_weight_sha256=f['models']['A']['weights_sha256'],J2_weight_sha256=f['models']['J2']['weights_sha256'],
 masked_cells=int((~valid).sum()),displayed_cells=int(valid.sum()),
 displayed_correction_percent_range=[float(100*(ratio[valid].min()-1)),float(100*(ratio[valid].max()-1))],
 caveat='No independently integrated two-dimensional NNLO reference is booked for this map. It shows where the fitted joint block changes event weights, not the accuracy of either distribution.')
np.savez_compressed(str(ROOT/'output/wm_joint_redistribution.npz'),y_edges=ye,c_edges=ce,count=n,A_probability=ha,J2_probability=hj,A_effective_count=eff,J2_over_A=ratio,display_mask=valid,metadata=np.array(json.dumps(meta)))
(ROOT/'output/wm_joint_redistribution.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
