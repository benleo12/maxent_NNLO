"""Prediction and multiplier spread across fixed training halves; no reference input."""
from pathlib import Path
import argparse,json
import numpy as np
from train import load_prior,load_training,prediction_columns,plain,write_json

ap=argparse.ArgumentParser();ap.add_argument("root",type=Path);a=ap.parse_args();r=a.root;summary={}
for charge in ("WP","WM"):
    ev,ids,meta=load_prior(r/"priors"/("w_prior_v6_nomec_"+charge.lower()+".npz"),charge)
    columns,labels,definition=prediction_columns(ev)
    cache=np.load(r/"predictions160"/charge/"predictions.npz",allow_pickle=False)
    cm=json.loads(str(cache["metadata"]));selected=cm["selected"];index=cm["models"].index(selected)
    outputs=[];lam=[]
    for subset in ("even","odd"):
        data=load_training(r/"pooled160"/("T_"+charge+".npz"),subset)
        z=np.load(r/"fits160"/charge/subset/("sequence_"+selected+".npz"),allow_pickle=False)
        outputs.append((z["weights"]@columns)*data.mean[0,:,None]);lam.append(z["lam"])
    difference=outputs[0]-outputs[1]
    expected=2*np.sqrt(np.maximum(np.diag(cache["covariance_numerical"][index]),0))
    scaled=np.divide(difference[0],expected,out=np.zeros(len(labels)),where=expected>0)
    summary[charge]=dict(labels=labels,half_prediction_difference=difference,
        linear_expected_error_of_half_difference=expected,
        central_half_difference_over_linear_error=scaled,
        max_abs_scaled_difference=float(abs(scaled).max()),
        multiplier_difference=lam[0]-lam[1],
        note="Same prior and feature set;2×full training-input error is a first-order equal-half expectation. No reference values are used and no extra selection rule is applied.")
write_json(r/"fits160/HALF_PREDICTION_STABILITY.json",summary)
print(json.dumps({ch:dict(max_abs_scaled_difference=v["max_abs_scaled_difference"]) for ch,v in summary.items()}))
