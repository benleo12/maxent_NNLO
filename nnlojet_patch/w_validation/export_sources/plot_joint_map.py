"""Redraw the W- joint-moment redistribution map from its compact cache."""
from pathlib import Path
import argparse,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('cache',type=Path);ap.add_argument('--output',type=Path);a=ap.parse_args()
z=np.load(str(a.cache),allow_pickle=False);meta=json.loads(str(z['metadata']));out=a.output or a.cache.parent;out.mkdir(parents=True,exist_ok=True)
r=100*(z['J2_over_A']-1);valid=z['display_mask'];lim=max(.5,float(np.ceil(np.max(np.abs(r[valid]))*2)/2))
plt.rcParams.update({'font.family':'serif','font.size':9,'mathtext.fontset':'stix','axes.labelsize':10,'axes.titlesize':9,'xtick.direction':'in','ytick.direction':'in'})
fig,ax=plt.subplots(figsize=(4.8,3.4));ax.set_facecolor('#ececec')
p=ax.pcolormesh(z['y_edges'],z['c_edges'],np.ma.array(r.T,mask=~valid.T),cmap='RdBu_r',vmin=-lim,vmax=lim,rasterized=True)
for edge in [-.7,-.6,.6,.7]:ax.axhline(edge,color='black',ls=':' if abs(edge)==.7 else '--',lw=.55,alpha=.65)
ax.set_xlabel(r'$|y_W|$');ax.set_ylabel(r'Signed $\cos\theta^*_{\rm CS}$');ax.set_title(r'$W^-:$ redistribution from rapidity-angle moments')
cb=fig.colorbar(p,ax=ax,pad=.025);cb.set_label(r'$100\,[P_{J2}/P_A-1]$ [\%]')
fig.text(.5,.015,'Same events; grey cells have fewer than 100 effective events',ha='center',fontsize=7)
fig.tight_layout(rect=(0,.035,1,1))
for ext in ['pdf','png']:fig.savefig(out/('wm_joint_redistribution.'+ext),dpi=220,bbox_inches='tight')
plt.close(fig)
(out/'wm_joint_redistribution_caption.txt').write_text('Redistribution in the W- sample when the six degree-two rapidity-angle moments are added to the production and angular constraints. The ratio uses exactly the same events in numerator and denominator, with the independently frozen J2 and A weights. Grey cells have fewer than 100 raw or A-weighted effective events. Dashed and dotted lines mark the flat-window and zero-window boundaries at |cos(theta*)|=0.6 and0.7. This is a correction map, not closure against an independent two-dimensional NNLO prediction.\n')
