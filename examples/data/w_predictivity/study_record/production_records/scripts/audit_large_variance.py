"""Inspect numerical validity of variance-dominating runs without reporting means."""
from pathlib import Path
import argparse,json,hashlib,re
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument("root",type=Path);ap.add_argument("--reference",type=Path);ap.add_argument("--output",type=Path)
a=ap.parse_args();r=a.root;reference=a.reference or r/"reference_p1"
audit=json.loads((reference/"PRECISION_ONLY.json").read_text())
target=audit["asymmetry"]["eta32"];total=sum(np.asarray(v)[:,0].sum() for v in target["channel_variance"].values())
records=[]
for charge in ("WP","WM"):
 z=np.load(reference/("V_"+charge+".npz"),allow_pickle=False)
 acc=z["acc"];errors=z["integration_errors"];all_seeds=z["seeds"]
 meta=json.loads(str(z["metadata"]));column=meta["labels"].index("eta32_00")
 for j,channel in enumerate(meta["channels"]):
  keep=all_seeds[j]>=0;values=acc[j,keep,column,0];seeds=all_seeds[j,keep]
  squares=(values-values.mean())**2;fraction=np.asarray(target["channel_variance"][charge])[j,0]/total
  if squares.sum()==0:continue
  for index in np.argsort(squares)[-5:][::-1]:
   directory=r/"runs/V"/charge/channel/("s%d"%seeds[index]);rec=json.loads((directory/"complete.json").read_text())
   log=(directory/"run.log").read_text(errors="replace")
   flags=[line for line in log.splitlines() if re.search(r"\b(warning|failed|failure|nan|infinite|unstable|exception|error stop)\b",line,re.I)]
   records.append(dict(charge=charge,channel=channel,seed=int(seeds[index]),
      fraction_of_total_asymmetry_variance=float(fraction*squares[index]/squares.sum()),
      returncode=rec["returncode"],completed=rec["ok"],elapsed_marker_present="Elapsed time" in log,
      all_raw_integrals_finite=bool(np.isfinite(acc[j,np.flatnonzero(keep)[index]]).all()),
      all_reported_integration_errors_finite=bool(np.isfinite(errors[j,np.flatnonzero(keep)[index]]).all()),
      warning_lines=flags,log_sha256=hashlib.sha256(log.encode()).hexdigest(),
      card_sha256=rec["card_sha256"],binary_sha256=rec["binary_sha256"],
      integration_error_for_this_bin_pb=float(errors[j,np.flatnonzero(keep)[index],column,0])))
records.sort(key=lambda row:row["fraction_of_total_asymmetry_variance"],reverse=True)
result=dict(observable="eta32 first-bin charge asymmetry",reference_central_predictions_blinded=True,
    records=records,top1_variance_fraction=records[0]["fraction_of_total_asymmetry_variance"],
    top5_variance_fraction=sum(row["fraction_of_total_asymmetry_variance"] for row in records[:5]),
    action="retain every run; no exclusion based on size or outcome; inspect flags as ordinary numerical validity evidence")
path=a.output or reference/"LARGE_VARIANCE_AUDIT.json";assert not path.exists();path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps(dict(top1_variance_fraction=result["top1_variance_fraction"],
                     top5_variance_fraction=result["top5_variance_fraction"],top_records=records[:5]),indent=2))
