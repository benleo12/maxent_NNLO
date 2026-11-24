"""Draw aggregate evidence only. Does not touch manuscript figure assets."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter

root=Path(__file__).resolve().parent
s=json.loads((root/'summary.json').read_text())
a=np.load(root/'diagnostics.npz')
oldsummary=root.parents[1]/'notes_2026-11-15/jet_binning_precision/summary.json'
colors=dict(prior='#666666',slope='#E49B2B',joint='#B5334B',reference='#172A3A',bins='#3877A7')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelsize':12,'axes.titlesize':12,'legend.fontsize':10,'savefig.dpi':180,
    'font.family':'DejaVu Sans','pdf.fonttype':42})

fig,axs=plt.subplots(1,3,figsize=(15.6,5.1),gridspec_kw={'width_ratios':[1.05,1,1.08]})
fig.subplots_adjust(left=.06,right=.985,bottom=.19,top=.83,wspace=.38)
edges=np.array([60,80,110,155,215,300,420,600,1000])
ratio=a['correction_ratio'].reshape(8,8).T
mask=a['common_mask'].reshape(8,8).T
ax=axs[0]; ax.set_facecolor('#f0f0f0')
arr=np.ma.masked_where(~mask,ratio)
mesh=ax.pcolormesh(edges,edges,arr,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=.65,vcenter=1,vmax=4.1),edgecolors='white',linewidth=.5)
for i,j in zip(*np.where(mask)):
    ax.text(np.sqrt(edges[j]*edges[j+1]),np.sqrt(edges[i]*edges[i+1]),f'{ratio[i,j]:.2f}',
        ha='center',va='center',fontsize=8,color='white' if ratio[i,j]>2.8 else '#202020')
ax.set(xscale='log',yscale='log',xlim=(60,1000),ylim=(60,1000),
       xlabel=r'Leading jet $p_T$ [GeV]',ylabel=r'Second jet $p_T$ [GeV]',
       title='Measured shape correction\nFixed order / shower prior')
for axis in [ax.xaxis,ax.yaxis]:
    axis.set_ticks([60,110,215,420,1000]); axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
cb=fig.colorbar(mesh,ax=ax,fraction=.047,pad=.025)
cb.set_label('Ratio of normalized bin probabilities',fontsize=10)

ax=axs[1]
selected=[(0,'Shower prior',colors['prior']),(1,'Two slope moments',colors['slope']),
          (9,'12 moments, including joint',colors['joint'])]
x=np.arange(2); offsets=[-.24,-.08,.08,.24]
for j,(i,label,color) in enumerate(selected):
    means=np.array([a['predictions'][i,7:9].sum(),a['predictions'][i,13:15].sum()])*100
    boot=np.stack([a['bootstrap'][:,i,7:9].sum(1),a['bootstrap'][:,i,13:15].sum(1)],axis=1)*100
    ax.errorbar(x+offsets[j],means,yerr=boot.std(0,ddof=1),fmt='o',color=color,
        markersize=6,capsize=3,label=label)
ref=np.array([a['reference'][7:9].sum(),a['reference'][13:15].sum()])*100
rb=np.stack([a['reference_bootstrap'][:,7:9].sum(1),a['reference_bootstrap'][:,13:15].sum(1)],axis=1)*100
ax.errorbar(x+offsets[3],ref,yerr=rb.std(0,ddof=1),fmt='s',color=colors['reference'],capsize=3,label='Independent fixed order')
ax.set(xticks=x,xticklabels=['125–250','250–500'],ylim=(0,46),
       xlabel=r'Leading jet $p_T$ [GeV]',ylabel=r'Fraction with $p_{T2}/p_{T1}>0.75$ [%]',
       title='The prior is missing balanced pairs')
ax.grid(axis='y',alpha=.18); ax.legend(loc='upper right',frameon=False,fontsize=9)

ax=axs[2]
ix=[1,2,3,4,9,8]
ks=[s['models'][i]['constraints'] for i in ix]
vals=a['metrics'][ix,1]*100
err=a['metrics_bootstrap'][:,ix,1].std(0,ddof=1)*100
ax.errorbar(ks,vals,yerr=err,fmt='o',color=colors['joint'],capsize=3,label='Moment fits')
prior=a['metrics'][0,1]*100
ax.axhline(prior,color=colors['prior'],ls=':',lw=1.4,label='Shower prior')
bs=json.loads(oldsummary.read_text())
entries=bs['models']
if isinstance(entries,dict): entries=list(entries.values())
hist=[r for r in entries if r['name'].startswith('merged_')]
ax.errorbar([r['constraints'] for r in hist],[100*r['tv2d_supported'] for r in hist],
    yerr=[100*r['tv2d_spread'] for r in hist],fmt='s--',color=colors['bins'],lw=1,capsize=3,label='Merged histogram fits')
ax.set(xlim=(0,30),ylim=(0,12.3),xlabel='Number of shape constraints',
       ylabel='2D shape discrepancy [%]',title='Few moments capture most of the correction')
ax.grid(axis='y',alpha=.18); ax.legend(frameon=False,loc='upper right',bbox_to_anchor=(1,.80),fontsize=9)
fig.suptitle(r'$Z+2$ jets: correcting the shower’s hard-radiation shape',fontsize=16,y=.98)
fig.text(.06,.045,'16,004 common prior events • LO two-parton reference • Fits use the original training inputs',fontsize=10,color='#444444')
fig.text(.06,.008,'Left and right: original 19-cell comparison region, covering 97.0% of the reference. Error bars: numerical resampling spread.',fontsize=10,color='#444444')
for ext in ('pdf','png'):
    fig.savefig(root/f'correction_evidence.{ext}',bbox_inches='tight')
plt.close(fig)

# Compact numbers for review, with paired resampling for summed probabilities.
rows=[]
for j,(i,label,color) in enumerate(selected):
    for k,sl in enumerate((slice(7,9),slice(13,15))):
        rr=a['reference'][sl].sum(); pred=a['predictions'][i,sl].sum()
        delta=a['bootstrap'][:,i,sl].sum(1)-a['reference_bootstrap'][:,sl].sum(1)
        rows.append(dict(method=label,leading_pt_interval=[[125,250],[250,500]][k],
            balanced_fraction=float(pred),fraction_sd=float(a['bootstrap'][:,i,sl].sum(1).std(ddof=1)),
            reference=float(rr),reference_sd=float(a['reference_bootstrap'][:,sl].sum(1).std(ddof=1)),
            difference=float(pred-rr),difference_sd=float(delta.std(ddof=1))))
(root/'physical_fractions.json').write_text(json.dumps(rows,indent=2)+'\n')
print(root/'correction_evidence.pdf')
