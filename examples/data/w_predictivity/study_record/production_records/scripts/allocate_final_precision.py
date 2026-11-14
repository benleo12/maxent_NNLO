"""One bounded final allocation, based only on reference variances and run costs."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.optimize import minimize

ap=argparse.ArgumentParser();ap.add_argument("audit",type=Path);ap.add_argument("--output",type=Path,required=True)
a=ap.parse_args();d=json.loads(a.audit.read_text());assert d["central_predictions_blinded"]
channels=["LO","R","V","RR","RV","VV"]
caps=np.tile([256,2048,256,2048,1024,128],2)
costs=np.tile([7.8,19.1,67.5,863.9,791.6,613.3],2)
counts=np.concatenate([d["charges"][charge]["n_runs"] for charge in ("WP","WM")])
rows=[];bounds=[];labels=[]
for ci,charge in enumerate(("WP","WM")):
    rec=d["charges"][charge]
    for i,(label,threshold) in enumerate(zip(rec["labels"],rec["thresholds"])):
        row=np.zeros(12);row[ci*6:ci*6+6]=np.asarray(rec["channel_variance"])[:,i]*counts[ci*6:ci*6+6]
        rows.append(row);bounds.append(threshold**2);labels.append(charge+"/"+label)
for group,rec in d["asymmetry"].items():
    for i in range(11):
        row=np.concatenate([np.asarray(rec["channel_variance"][charge])[:,i]*counts[ci*6:ci*6+6]
                            for ci,charge in enumerate(("WP","WM"))])
        rows.append(row);bounds.append(rec["threshold"]**2);labels.append(group+"/%02d"%i)
matrix=np.asarray(rows);goals=np.asarray(bounds)
# Aim for10% error margin. Where even the declared caps cannot reach that,
# stay within5% of the best capped precision rather than demand an impossible
# constraint. Such forecast misses remain explicit and do not alter goals.
minimum=matrix@(1/caps);target=np.maximum(.9**2*goals,1.05**2*minimum)
def objective(logn):return float(costs@np.exp(logn)/1e6)
def objective_jac(logn):return costs*np.exp(logn)/1e6
def constraint(logn):return 1-(matrix@np.exp(-logn))/target
def constraint_jac(logn):return matrix*np.exp(-logn)[None,:]/target[:,None]
fit=minimize(objective,np.log(caps),jac=objective_jac,bounds=list(zip(np.log(counts),np.log(caps))),
             constraints=[dict(type="ineq",fun=constraint,jac=constraint_jac)],method="SLSQP",
             options=dict(ftol=1e-10,maxiter=1000))
if not fit.success:raise RuntimeError(fit.message)
chosen=np.minimum(caps,np.maximum(counts,np.ceil(np.exp(fit.x)).astype(int)))
forecast=np.sqrt(matrix@(1/chosen));initial=np.sqrt(matrix@(1/counts))
record=dict(central_predictions_blinded=True,final_extension=True,counts={},caps={},
    previous_counts={},labels=labels,current_errors=initial.tolist(),forecast_errors=forecast.tolist(),
    thresholds=np.sqrt(goals).tolist(),forecast_passed=(forecast<=np.sqrt(goals)).tolist(),
    variance_model="each observed channel variance scales as current_count/new_count; all retained equal-budget runs",
    allocation="minimum measured CPU cost with10% error margin where feasible; capped unattainable goals remain explicit",
    additional_runs=int((chosen-counts).sum()),cumulative_runs=int(chosen.sum()),
    cumulative_points=int(chosen.sum()*250000),additional_cpu_hours=float(costs@(chosen-counts)/3600),
    optimizer_message=fit.message)
for i,charge in enumerate(("WP","WM")):
    record["counts"][charge]=dict(zip(channels,chosen[i*6:i*6+6].tolist()))
    record["caps"][charge]=dict(zip(channels,caps[i*6:i*6+6].tolist()))
    record["previous_counts"][charge]=dict(zip(channels,counts[i*6:i*6+6].tolist()))
assert not a.output.exists();a.output.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
print(json.dumps(record,indent=2))
