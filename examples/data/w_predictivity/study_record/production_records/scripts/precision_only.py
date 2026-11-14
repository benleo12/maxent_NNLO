"""Reference precision only: deliberately never print central predictions."""
from pathlib import Path
import argparse
import json
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument("directory",type=Path);ap.add_argument("--output",type=Path,required=True)
a=ap.parse_args();data={};summary=dict(central_predictions_blinded=True,charges={},asymmetry={})
for charge in ("WP","WM"):
    z=np.load(a.directory/("V_"+charge+".npz"),allow_pickle=False)
    meta=json.loads(str(z["metadata"])); labels=meta["labels"]; mean=z["mean"][:,0]
    channel_cov=[]
    for acc,seeds in zip(z["acc"],z["seeds"]):
        group=acc[seeds>=0,:,0]
        channel_cov.append(np.cov(group,rowvar=False,ddof=1)/len(group))
    channel_cov=np.asarray(channel_cov)
    numerator=np.r_[np.arange(1,7),4];denominator=np.r_[np.zeros(6,dtype=int),3]
    jac=np.zeros((7,len(labels)));jac[np.arange(7),numerator]=1/mean[denominator]
    jac[np.arange(7),denominator]-=mean[numerator]/mean[denominator]**2
    variance=np.array([np.einsum('ij,jk,ik->i',jac,c,jac) for c in channel_cov])
    errors=np.sqrt(np.maximum(variance.sum(0),0));thresholds=np.array([.0005]*6+[.002])
    summary["charges"][charge]=dict(n_runs=z["n_runs"].tolist(),channels=meta["channels"],
        labels=labels[1:7]+["pt32_given_hard"],errors=errors.tolist(),thresholds=thresholds.tolist(),
        passed=(errors<=thresholds).tolist(),channel_variance=variance.tolist())
    data[charge]=(mean,channel_cov,labels)
for group in ("eta28","eta32","eta_mt60"):
    ix=[i for i,label in enumerate(data["WP"][2]) if label.startswith(group+"_")]
    p,m=data["WP"][0][ix],data["WM"][0][ix];total=p+m
    derivatives=dict(WP=2*m/total**2,WM=-2*p/total**2)
    variance={charge:(np.diagonal(data[charge][1],axis1=1,axis2=2)[:,ix]*derivatives[charge]**2)
              for charge in ("WP","WM")}
    errors=np.sqrt(sum(value.sum(0) for value in variance.values()))
    summary["asymmetry"][group]=dict(errors=errors.tolist(),threshold=.002,
        passed=(errors<=.002).tolist(),channel_variance={key:value.tolist() for key,value in variance.items()})
summary["all_precision_passed"]=all(all(v["passed"]) for v in summary["charges"].values()) and all(all(v["passed"]) for v in summary["asymmetry"].values())
assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
print(json.dumps(dict(all_precision_passed=summary["all_precision_passed"],
    charges={key:dict(errors=value["errors"],passed=value["passed"]) for key,value in summary["charges"].items()},
    asymmetry={key:dict(errors=value["errors"],passed=value["passed"]) for key,value in summary["asymmetry"].items()})))
