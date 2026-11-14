"""Frozen full-prior predictions and paired covariance, without reference reads."""
from pathlib import Path
import argparse
import gc
import json
import sys
import time
import numpy as np
from train import (ROOT, CANDIDATES, load_prior, load_training, prediction_columns,
                   sha, plain, write_json)
from maxent_upgrade import fit_constraints, load_frozen, plan_observable
from w_physical_inputs import build_blocks, candidate_blocks


def cross_covariance(first, second):
    numeric=np.zeros((len(first.labels),len(second.labels)))
    prior=np.zeros_like(numeric)
    for key,value in first.sources.items():
        if key in second.sources:
            other=second.sources[key]
            assert np.array_equal(value["covariance"],other["covariance"])
            numeric+=value["jacobian"]@value["covariance"]@other["jacobian"].T
    for key,value in first.priors.items():
        if key in second.priors:
            prior+=value["influence"].T@second.priors[key]["influence"]
    return numeric,prior


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--charge",choices=["WP","WM"],required=True);ap.add_argument("--runs",type=int,default=160)
    a=ap.parse_args();root=a.root;charge=a.charge;start=time.monotonic()
    fits=root/("fits%d"%a.runs);review=json.loads((fits/"TRAINING_REVIEW.json").read_text())
    assert review["ready_to_freeze"]
    selection=json.loads((fits/charge/"full/SELECTION.json").read_text());selected=selection["selected"]
    assert review["charges"][charge]["stable"] or a.runs>=640
    events,ids,prior_meta=load_prior(root/"priors"/("w_prior_v6_nomec_"+charge.lower()+".npz"),charge)
    data=load_training(root/("pooled%d"%a.runs)/("T_"+charge+".npz"))
    blocks=build_blocks(events,data,charge);values,labels,definition=prediction_columns(events)
    prior_id="w-predictivity-full-prior-"+charge.lower();dest=root/("predictions%d"%a.runs)/charge;dest.mkdir(parents=True,exist_ok=True)
    models=["P"]+list(CANDIDATES)+[selected];assert len(set(models))==len(models)
    predictions={};archives={};summaries={}
    for name in models:
        if name=="P":
            file=dest/"prior_frozen.npz"
            if file.exists():
                fit=load_frozen(file,events["weight"],data,[],prior_id=prior_id,event_ids=ids)
            else:
                fit=fit_constraints(events["weight"],data,[],prior_id=prior_id,event_ids=ids,l2=1e-4)
                fit.freeze(file,selection=dict(model="P",definition="unchanged LO+PS shape with coherent NNLO baseline rates",
                           prior=prior_meta,validation_status="unopened; fixed baseline comparison"),
                           validation_plan={"all":plan_observable(values,labels=labels,definition=definition)})
        else:
            file=fits/charge/"full"/(("sequence_" if name==selected else "candidate_")+name+".npz")
            fit=load_frozen(file,events["weight"],data,candidate_blocks(blocks,charge,name),
                            prior_id=prior_id,event_ids=ids)
        predictions[name]=fit.predict(values,labels=labels,definition=definition,validation_key="all")
        archives[name]=dict(path=str(file),sha256=sha(file),freeze_id=fit.frozen_id)
        summaries[name]=dict(rate_pb=data.mean[0].tolist(),
            ess_fraction=float(1/(len(fit.p)*(fit.weights@fit.weights))),
            q_over_p_range=[float(np.min(fit.weights/fit.p)),float(np.max(fit.weights/fit.p))])
        del fit;gc.collect()
        print(json.dumps(dict(stage="prediction",charge=charge,model=name,elapsed_s=time.monotonic()-start)),flush=True)
    means=np.array([predictions[name].mean for name in models])
    numeric=np.array([predictions[name].covariance_numerical for name in models])
    prior=np.array([predictions[name].covariance_prior for name in models])
    pairs=[("R","anchor"),("anchor","P")]
    pairs += [(name,"R" if name.startswith("R+") or name in {"H0","H","S0","S"} else "anchor")
              for name in CANDIDATES if name not in {"anchor","R"}]
    pairs += [("N","G"),("W","G"),("R+N","R+G"),("R+W","R+G"),("H","H0"),("S","S0")]
    pairs += [(selected,name) for name in ("P","anchor","R","H","S")]
    cross_num=[];cross_prior=[]
    for first,second in pairs:
        n,p=cross_covariance(predictions[first],predictions[second]);cross_num.append(n);cross_prior.append(p)
    metadata=dict(charge=charge,models=models,selected=selected,pairs=pairs,labels=labels,definition=definition,
        scale_names=data.scale_names,training_input_identity=data.identity,prior=prior_meta,frozen_archives=archives,
        model_diagnostics=summaries,uncertainty="linear fitted-response common-input covariance plus independent weighted-row prior influence; scales separate",
        validation_reference_opened=False,elapsed_s=time.monotonic()-start)
    target=dest/"predictions.npz";assert not target.exists()
    np.savez_compressed(target,means=means,covariance_numerical=numeric,covariance_prior=prior,
        cross_covariance_numerical=np.array(cross_num),cross_covariance_prior=np.array(cross_prior),metadata=np.array(json.dumps(plain(metadata),allow_nan=False)))
    write_json(dest/"PREDICTIONS_COMPLETE.json",dict(sha256=sha(target),models=models,selected=selected,
        elapsed_s=time.monotonic()-start,validation_reference_opened=False))
    print(json.dumps(dict(stage="complete",charge=charge,elapsed_s=time.monotonic()-start)),flush=True)


if __name__=="__main__":main()
