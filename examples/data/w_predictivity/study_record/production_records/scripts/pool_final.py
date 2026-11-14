"""Pool the single final extension and record the finite stop before unblinding."""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys,time
import numpy as np
from aggregate import aggregate

ap=argparse.ArgumentParser();ap.add_argument("root",type=Path);a=ap.parse_args();r=a.root
plan=json.loads((r/"FINAL_PRECISION_ALLOCATION.json").read_text())
dest=r/"reference_final";dest.mkdir(exist_ok=True);counts={};hashes={}
for charge in ("WP","WM"):
    result=aggregate(r,"V",charge,64,channel_counts=plan["counts"])
    expected=list(plan["counts"][charge][key] for key in ("LO","R","V","RR","RV","VV"))
    assert np.array_equal(result["n_runs"],expected)
    path=dest/("V_"+charge+".npz");assert not path.exists();np.savez_compressed(path,**result)
    counts[charge]=dict(zip(("LO","R","V","RR","RV","VV"),result["n_runs"].tolist()))
    hashes[charge]=hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(dict(charge=charge,counts=counts[charge],sha256=hashes[charge])),flush=True)
subprocess.check_call([sys.executable,str(r/"scripts/precision_only.py"),str(dest),"--output",str(dest/"PRECISION_ONLY.json")])
subprocess.check_call([sys.executable,str(r/"scripts/audit_large_variance.py"),str(r),"--reference",str(dest)])
audit=json.loads((dest/"PRECISION_ONLY.json").read_text())
record=dict(final_stop=True,central_predictions_blinded=True,extensions_used=2,
    completed_counts=counts,reference_sha256=hashes,total_reference_runs=sum(sum(c.values()) for c in counts.values()),
    total_reference_points=250000*sum(sum(c.values()) for c in counts.values()),
    absolute_cap_points=2880000000,all_precision_passed=audit["all_precision_passed"],
    stopping_record_sha256=hashlib.sha256((r/"scripts/PRECISION_STOPPING.md").read_bytes()).hexdigest(),
    final_allocation_sha256=hashlib.sha256((r/"FINAL_PRECISION_ALLOCATION.json").read_bytes()).hexdigest(),
    decision="Stop after the single final extension; retain all physical seeds and report every remaining precision miss, with no model or bin changes.",
    recorded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
assert record["total_reference_points"]==plan["cumulative_points"]<=record["absolute_cap_points"]
(dest/"PRECISION_STOPPING_COMPLETE.json").write_text(json.dumps(record,indent=2)+"\n")
print("FINAL_REFERENCE_PRECISION_REVIEW_READY",flush=True)
