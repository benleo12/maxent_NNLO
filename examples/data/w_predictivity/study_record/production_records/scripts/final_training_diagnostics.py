"""Descriptive diagnostics after the single-pass selection, with no retuning."""
from pathlib import Path
import argparse,json
from train import load_prior,load_training,diagnostic_record,write_json
from maxent_upgrade import load_frozen
from w_physical_inputs import build_blocks,candidate_blocks

ap=argparse.ArgumentParser();ap.add_argument("root",type=Path);a=ap.parse_args();r=a.root;out={}
for charge in ("WP","WM"):
    events,ids,meta=load_prior(r/"priors"/("w_prior_v6_nomec_"+charge.lower()+".npz"),charge)
    data=load_training(r/"pooled160"/("T_"+charge+".npz"));blocks=build_blocks(events,data,charge)
    selection=json.loads((r/"fits160"/charge/"full/SELECTION.json").read_text());name=selection["selected"]
    fit=load_frozen(r/"fits160"/charge/"full"/("sequence_"+name+".npz"),events["weight"],data,
        candidate_blocks(blocks,charge,name),prior_id="w-predictivity-full-prior-"+charge.lower(),event_ids=ids)
    out[charge]=dict(selected=name,descriptive_only=True,selection_changed=False,
        diagnostics={group:diagnostic_record(fit.diagnose(blocks[group])) for group in ("D","M")})
write_json(r/"fits160/FINAL_TRAINING_DIAGNOSTICS.json",out)
print(json.dumps({ch:{group:value["test"] for group,value in rec["diagnostics"].items()} for ch,rec in out.items()}))
