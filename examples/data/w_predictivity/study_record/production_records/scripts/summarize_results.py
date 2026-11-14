"""Compact reader-facing numbers extracted from the complete validated report."""
from pathlib import Path
import argparse,json

ap=argparse.ArgumentParser();ap.add_argument("report",type=Path);a=ap.parse_args()
d=json.loads(a.report.read_text());out=dict(charges={},asymmetry={},precision=d["precision"],
    uncertainty_sources=d["uncertainty_sources"],scale_convention=d["scale_convention"])
def entry(summary,index,factor=100):
    return dict(value=factor*summary["central"][index],error=factor*summary["numerical_error"][index])
for charge,rec in d["charges"].items():
    labels=rec["metric_labels"];selected=rec["selected"];models=["P","anchor","R","H",selected]
    c=dict(selected=selected,units="percent for acceptances; percentage points for differences",
           acceptance={},old_cut_recoil_ledger={},model_diagnostics=rec["model_diagnostics"])
    for test in ("old_pt27","old_pt30","pt28","pt32","mt60","combined","pt32hard","pt32_given_hard"):
        i=labels.index(test)
        c["acceptance"][test]=dict(predictions={name:entry(rec["models"][name]["prediction"],i) for name in models},
            reference=entry(rec["reference"],i),
            selected_minus_reference=entry(rec["models"][selected]["difference_from_reference"],i),
            paired_shifts={pair:entry(rec["paired"][pair],i) for pair in
                          ("R minus anchor","H minus R",selected+" minus H",selected+" minus R")})
    for cut in (27,30):
        rows=[]
        for region in range(3):
            row=dict(recoil_region=region)
            for model in ("R",selected):
                row[model]={component:entry(rec["models"][model]["difference_from_reference"],
                     labels.index("old%d_qt%d_%s"%(cut,region,component)))
                     for component in ("projected","migration","physical")}
            rows.append(row)
        c["old_cut_recoil_ledger"][str(cut)]=rows
    out["charges"][charge]=c
selected=d["charges"]["WP"]["selected"]
assert selected==d["charges"]["WM"]["selected"]
for name in ("eta28","eta32","eta_mt60"):
    rows=[]
    for j in range(11):
        i=d["asymmetry"]["labels"].index("%s_%02d"%(name,j))
        row=dict(bin=j,model_minus_reference={model:entry(d["asymmetry"]["models"][model]["difference_from_reference"],i,1)
                                             for model in ("P","anchor","R","H",selected)},
                 spin_shift=entry(d["asymmetry"]["paired"][selected+" minus R"],i,1))
        rows.append(row)
    out["asymmetry"][name]=rows
print(json.dumps(out,indent=2,allow_nan=False))
