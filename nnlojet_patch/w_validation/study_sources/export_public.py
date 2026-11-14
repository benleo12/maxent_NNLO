"""Export training-only public inputs separately from independent validation caches."""
from pathlib import Path
import json,shutil,math
import numpy as np
from train import ROOT,MODELS,L2,sha

out=ROOT/'output'/'public';out.mkdir(parents=True,exist_ok=True)
def strict(value):
    if isinstance(value,dict):return {k:strict(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [strict(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value

def write_json(path,value):
    path.write_text(json.dumps(strict(value),indent=2,allow_nan=False)+'\n')
for charge in ['WP','WM']:
    if not (ROOT/'output'/charge/'MODEL_FREEZE.json').exists():continue
    fr=json.loads((ROOT/'output'/charge/'MODEL_FREEZE.json').read_text())
    z=np.load(str(ROOT/'input'/('T_%s.npz'%charge)),allow_pickle=False)
    meta=json.loads(str(z['metadata']))
    assert np.isfinite(z['mean']).all() and np.isfinite(z['cov']).all(), 'Nonfinite physical input cannot be published'
    configs={}
    for model in MODELS:
        cfg=dict(born={'m_w':{'range':[40.,1000.],'map':['bw',80.385,2.085]},
                       'y_abs':{'range':[0.,4.],'map':'lin'}},
            recoil={'pt_w':{'range':[25.,500.],'map':'log','soft_lo':25.,
                            'profile':{'a':25.,'b':50.,'c':500.},'transition':'blend'}},
            fixed_orders=dict(fr['orders']),L2=L2)
        if model!='B':
            cfg['recoil']['ctw1']=dict(range=[0.,2.],map='lin',soft_lo=0.,
                profile={'a':.3,'b':.4,'c':1.6,'d':1.7},transition='windowed')
            cfg['fixed_orders']['ctw1']=6
        degree={'B':0,'A':0,'J2':2,'J4':4}[model]
        if degree:
            cfg['mixed']={'rapidity_angle':dict(observables=['y_abs','ctw1'],
                range={'y_abs':[0.,4.],'ctw1':[0.,2.]},map={'y_abs':'lin','ctw1':'lin'},
                profile={'ctw1':{'a':.3,'b':.4,'c':1.6,'d':1.7}},n=degree,include_rate=False)}
        configs[model]=cfg
    info=dict(schema='independent-w-prediction-study-v1',process=charge,order='NNLO QCD',
        sqrts_GeV=7000.,pdf='CT18NNLO member 0',scale_base='m_lnu',
        scale_factors=[[1,1],[2,2],[.5,.5],[2,1],[1,2],[.5,1],[1,.5]],
        electroweak={'scheme':'Gmu/CMS','MW':80.385,'GammaW':2.085,'MZ':91.1876,'GammaZ':2.4952,'CKM':'FULL'},
        baseline_cuts={'pt_l_min_GeV':20.,'pt_nu_min_GeV':25.,'mt_min_GeV':40.,'abs_eta_l_max':2.5},
        leptons='Born-level charged leptons, no QED radiation',
        angle='ctw1 = 1 + sign(pz_W) [(El+pzl)(Enu-pznu)-(El-pzl)(Enu+pznu)]/[m_lnu sqrt(m_lnu^2+qT^2)]',
        joint_definition='w(ctw1) T_k(|yW|/2-1) T_n(ctw1-1), k=1..D, n=0..D. n=0 terms retained.',
        recoil_definition='Directly booked T_n of log(pT) mapped on [25,500]; profile (25,50,500). Do not apply old [25,2500] basis transformation.',
        accumulator_labels=meta['labels'],accumulator_mean_pb=z['mean'].tolist(),
        accumulator_covariance_pb2=z['cov'].tolist(),
        uncertainty_definition='Covariance of independent equal-budget seed means, summed over independent channels. Ratios use their common denominator and full covariance. Scale vectors are coherent alternatives, not Gaussian errors.',
        channels=meta['channels'],replicas_per_channel=meta['replicas'],points_per_run=meta['points_per_run'],
        seeds=z['seeds'].tolist(),selected=fr['selected'],configs=configs,
        training_input_sha256=fr['input_sha256'],model_freeze_utc=fr['freeze_utc'],
        training_diagnostics=fr['models'],prior_identity=fr['prior'],
        validation_independent=True,validation_file='w_%s_validation.npz'%charge.lower())
    write_json(out/('w_%s_training.json'%charge.lower()),info)
    # Preserve every seed accumulator for coherent channel-wise resampling.
    np.savez_compressed(str(out/('w_%s_training_runs.npz'%charge.lower())),
                        acc=z['acc'],seeds=z['seeds'],labels=np.array(meta['labels']),channels=np.array(meta['channels']))
    predictions=ROOT/'output'/charge/'predictions.npz'
    if predictions.exists():
        shutil.copy2(predictions,out/('w_%s_validation.npz'%charge.lower()))
        vz=np.load(str(ROOT/'input'/('V_%s.npz'%charge)),allow_pickle=False)
        vm=json.loads(str(vz['metadata']))
        np.savez_compressed(str(out/('w_%s_validation_runs.npz'%charge.lower())),
            acc=vz['acc'],seeds=vz['seeds'],labels=np.array(vm['labels']),channels=np.array(vm['channels']),
            n_runs=np.sum(vz['seeds']>=0,axis=1),metadata=vz['metadata'])
    report=ROOT/'output'/charge/'validation.json'
    if report.exists():write_json(out/('w_%s_validation.json'%charge.lower()),json.loads(report.read_text()))
shutil.copy2(ROOT/'scripts/load_study.py',out/'load_study.py')
shutil.copy2(ROOT/'scripts/PROTOCOL.txt',out/'PROTOCOL.txt')
for name in ['RR_EXTENSION_PROTOCOL.txt','R_RV_EXTENSION_PROTOCOL.txt','FINAL_PRECISION_PROTOCOL.txt','CACHE_AUDIT.txt','PDF_EQUIVALENCE.txt','JACOBIAN_CHECK.json']:
    if (ROOT/'scripts'/name).exists():shutil.copy2(ROOT/'scripts'/name,out/name)
if (ROOT/'booking').exists():
    shutil.copytree(ROOT/'booking',out/'booking',dirs_exist_ok=True)
shutil.copy2(ROOT/'scripts/plot_predictions.py',out/'plot_predictions.py')
if (ROOT/'output/charge_asymmetry.json').exists():
    write_json(out/'charge_asymmetry.json',json.loads((ROOT/'output/charge_asymmetry.json').read_text()))
for name in ['wm_joint_redistribution.npz','wm_joint_redistribution.json','VALIDATION_COMPLETE.json','RESULTS_SUMMARY.json','FINAL_MANIFEST.json','MODEL_DIFFERENCE_CHECK.json']:
    if (ROOT/'output'/name).exists():shutil.copy2(ROOT/'output'/name,out/name)
shutil.copy2(ROOT/'scripts/plot_joint_map.py',out/'plot_joint_map.py')
sources=out/'study_sources';sources.mkdir(exist_ok=True)
for name in ['train.py','validate.py','export_public.py','stat_utils.py','aggregate.py',
             'check_jacobian.py','technical_checks.py','make_joint_map.py','precision_only.py',
             'precision_sources.py','allocate_precision.py','summarize_results.py','make_final_manifest.py','check_model_differences.py','pool_from_base.py','compare_pooling.py']:
    if (ROOT/'scripts'/name).exists():shutil.copy2(ROOT/'scripts'/name,sources/name)
for ch in ['WP','WM']:
    if (ROOT/'output'/ch/'MODEL_FREEZE.json').exists():
        shutil.copy2(ROOT/'output'/ch/'MODEL_FREEZE.json',sources/('MODEL_FREEZE_%s.json'%ch))
for name in ['precision_only.json','precision_sources.json']:
    if (ROOT/'input'/name).exists():shutil.copy2(ROOT/'input'/name,sources/('FINAL_'+name))
(sources/'README.txt').write_text('Exact authors\' study sources for auditing selection, covariance and validation. These scripts ran under /Users/benassi/reweighting/w_validation_20261113/scripts, with fixed prior and package snapshot paths. They are not the portable public fitting interface. Use the separately documented public runner and compact-cache plotters for ordinary reproduction. train.py opens training inputs only; validate.py requires both freeze records before opening independent references.\n')
if (ROOT/'software_checks').exists():shutil.copytree(ROOT/'software_checks',sources/'software_checks',dirs_exist_ok=True)
(sources/'SHA256SUMS').write_text('\n'.join(sha(p)+'  '+str(p.relative_to(sources)) for p in sorted(sources.rglob('*')) if p.is_file() and p.name!='SHA256SUMS')+'\n')
(out/'SHA256SUMS').write_text('\n'.join(sha(p)+'  '+p.name for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS')+'\n')
print(out)
