import sys, numpy as np, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maxent_upgrade import upgrade, compute_fo_moments
rng=np.random.default_rng(2)
# prior events
Np=150000
prior=dict(mll=rng.normal(91,3,Np).clip(66,116), y_abs=np.abs(rng.normal(0,1.1,Np)).clip(0,2.4),
           pT_ll=rng.exponential(8,Np).clip(0.5,200), weight=np.ones(Np))
# FO events (independent), with a real recoil-tail lift + 7 toy scale weights
Nf=120000
foM=rng.normal(91,3,Nf).clip(66,116); foY=np.abs(rng.normal(0,1.1,Nf)).clip(0,2.4)
foPT=rng.exponential(8,Nf).clip(0.5,200)
w=1.0+0.4*(foPT/100)          # NNLO lifts the recoil tail
wsc=np.array([w*f for f in (1.0,1.12,0.9,1.08,0.92,1.05,0.96)])  # 7 scales
fo_events=dict(mll=foM, y_abs=foY, pT_ll=foPT, weight=w, weight_scales=wsc)
cfg=dict(born={'mll':{'range':(66.,116.),'map':'lin'},'y_abs':{'range':(0.,2.4),'map':'lin'}},
         recoil={'pT_ll':{'range':(0.5,200.),'map':'log','soft_lo':0.5}})
moments=compute_fo_moments(fo_events, cfg, x_match=10.0, x_hi=200.0)
print("recoil window moments (event-level):", [round(v,3) for v in moments['recoil']['pT_ll']['window_values'][:6]])
print("recoil window errors:", [round(v,4) for v in moments['recoil']['pT_ll']['window_errors'][:6]])
print("rate=%.4f"%moments['recoil']['pT_ll']['rate'])
t0=time.time(); r=upgrade(prior, moments, cfg); dt=time.time()-t0
print("MOMENT-PATH RESULT:", r.summary(), "solve=%.1fs"%dt)
print("chosen:", r.chosen_moments)
print("recoil SNR:", [round(float(s),1) for s in r.moment_snr['pT_ll']])
assert (r.weights>0).all()
print("OK positive, event-level moment interface end-to-end")
