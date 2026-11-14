"""Independent reference comparison after immutable training choices and precision audit."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np


def plain(v):
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if isinstance(v,np.ndarray):return plain(v.tolist())
    if isinstance(v,np.generic):return plain(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def mapped(mean,cov,rows,denominators):
    num=mean@rows.T;den=mean[:,denominators]
    assert (den>0).all()
    values=num/den
    jac=rows/den[0,:,None]
    jac[np.arange(len(rows)),denominators]-=num[0]/den[0]**2
    return values,jac@cov@jac.T,jac


def metrics(labels):
    ix={k:i for i,k in enumerate(labels)};rows=[];names=[];den=[]
    def add(name,terms,denominator="norm_born"):
        row=np.zeros(len(labels))
        for key,coefficient in terms.items():row[ix[key]]+=coefficient
        names.append(name);rows.append(row);den.append(ix[denominator])
    for cut in ("pt28","pt32","hard","pt32hard","mt60","combined"):
        add(cut,{"hold_"+cut:1})
    add("pt32_given_hard",{"hold_pt32hard":1},"hold_hard")
    for cut in (27,30):
        add("old_pt%d"%cut,{"expl_qc_%d_%d_pt%d"%(q,c,cut):1 for q in range(3) for c in range(3)})
        for axis in ("qt","angle"):
            for region in range(3):
                cells=[(q,c) for q in range(3) for c in range(3) if (q if axis=="qt" else c)==region]
                physical={"expl_qc_%d_%d_pt%d"%(q,c,cut):1 for q,c in cells}
                projected={"expl_qc_%d_%d_proj%d"%(q,c,cut):1 for q,c in cells}
                migration=dict(physical);migration.update({key:-1 for key in projected})
                for component,terms in (("physical",physical),("projected",projected),("migration",migration)):
                    add("old%d_%s%d_%s"%(cut,axis,region,component),terms)
    for label in labels:
        if label.startswith("expl_"):add(label,{label:1})
    return names,np.asarray(rows),np.asarray(den)


def summarize(mean,cov):
    error=np.sqrt(np.maximum(np.diag(cov),0))
    return dict(central=mean[0],numerical_error=error,coherent_scales=mean,
                scale_lower=mean.min(0),scale_upper=mean.max(0))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("root",type=Path);ap.add_argument("--runs",type=int,default=160)
    ap.add_argument("--reference",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--allow-precision-misses",action="store_true")
    a=ap.parse_args();r=a.root;freeze=json.loads((r/("fits%d"%a.runs)/"FINAL_TRAINING_FREEZE.json").read_text())
    precision=json.loads((a.reference/"PRECISION_ONLY.json").read_text())
    scale_evidence=json.loads((r/"production_training/SCALE_ORDER_AUDIT.json").read_text())
    expected_scales=[[1,1],[2,2],[.5,.5],[2,1],[1,2],[.5,1],[1,.5]]
    assert scale_evidence["all_actual_cards_match"] and scale_evidence["scale_factors_muR_muF"]==expected_scales
    if not precision["all_precision_passed"]:
        assert a.allow_precision_misses,"a failed precision criterion requires the explicit finite stopping record"
        stop=json.loads((a.reference/"PRECISION_STOPPING_COMPLETE.json").read_text())
        assert stop["final_stop"] and stop["central_predictions_blinded"]
    assert not a.output.exists();a.output.mkdir(parents=True)
    report=dict(training_freeze=freeze,reference_directory=str(a.reference),precision=precision,charges={},
        uncertainty_sources="training NNLO integration with fit response; finite weighted prior with refit response; independent reference integration for residuals; first-order covariance",
        scale_convention="all differences subtract the corresponding coherent scale member on both sides; separate absolute model/reference scale arrays are retained",
        scale_factors_muR_muF=[[1,1],[2,2],[.5,.5],[2,1],[1,2],[.5,1],[1,.5]],
        central_scale="charged-lepton plus neutrino invariant mass")
    full={}
    for charge in ("WP","WM"):
        z=np.load(r/("predictions%d"%a.runs)/charge/"predictions.npz",allow_pickle=False)
        meta=json.loads(str(z["metadata"]));v=np.load(a.reference/("V_"+charge+".npz"),allow_pickle=False)
        vm=json.loads(str(v["metadata"]));assert vm["group"]=="V" and vm["labels"]==meta["labels"]
        assert vm["scale_factors_muR_muF"]==expected_scales
        assert meta["scale_names"]==scale_evidence["scale_names"]
        train_ids=set(meta["training_input_identity"]["run_ids"])
        actual_seeds=v["seeds"][v["seeds"]>=0]
        assert len(np.unique(actual_seeds))==len(actual_seeds)
        assert np.array_equal((v["seeds"]>=0).sum(axis=1),v["n_runs"])
        ref_ids={"nnlojet:%d"%seed for seed in actual_seeds}
        assert not train_ids&ref_ids
        assert meta["selected"]==freeze["selection"][charge]["selected"]
        assert meta["frozen_archives"][meta["selected"]]["sha256"]==freeze["selection"][charge]["sha256"]
        expected_eta=["%s_%02d"%(group,i) for group in ("eta28","eta32","eta_mt60") for i in range(11)]
        assert meta["labels"][7:40]==expected_eta
        labels,rows,den=metrics(meta["labels"]);refmean=v["mean"].T;refcov=v["cov"]
        refvalue,refmetriccov,refjac=mapped(refmean,refcov,rows,den)
        means=z["means"];num=z["covariance_numerical"];prior=z["covariance_prior"]
        values=[];covn=[];covp=[];jacs=[];models={}
        for i,name in enumerate(meta["models"]):
            value,n,jac=mapped(means[i],num[i],rows,den);p=jac@prior[i]@jac.T
            values.append(value);covn.append(n);covp.append(p);jacs.append(jac)
            models[name]=dict(prediction=summarize(value,n+p),
                difference_from_reference=summarize(value-refvalue,n+p+refmetriccov))
        values=np.asarray(values);covn=np.asarray(covn);covp=np.asarray(covp)
        paired={};pairedn=[];pairedp=[]
        for k,(first,second) in enumerate(meta["pairs"]):
            i,j=meta["models"].index(first),meta["models"].index(second)
            n=jacs[i]@z["cross_covariance_numerical"][k]@jacs[j].T
            p=jacs[i]@z["cross_covariance_prior"][k]@jacs[j].T
            dn=covn[i]+covn[j]-n-n.T;dp=covp[i]+covp[j]-p-p.T
            pairedn.append(dn);pairedp.append(dp)
            paired[first+" minus "+second]=summarize(values[i]-values[j],dn+dp)
        report["charges"][charge]=dict(selected=meta["selected"],models=models,
            reference=summarize(refvalue,refmetriccov),paired=paired,metric_labels=labels,
            event_count=meta["prior"]["n_fid"],training_and_reference_seeds_disjoint=True,
            model_diagnostics=meta["model_diagnostics"])
        np.savez_compressed(a.output/(charge+"_metrics.npz"),means=values,covariance_numerical=covn,
            covariance_prior=covp,reference=refvalue,reference_covariance=refmetriccov,
            paired_covariance_numerical=np.asarray(pairedn),paired_covariance_prior=np.asarray(pairedp),
            metadata=np.array(json.dumps(dict(labels=labels,models=meta["models"],pairs=meta["pairs"],selected=meta["selected"]))))
        full[charge]=dict(meta=meta,means=means,numeric=num,prior=prior,refmean=refmean,refcov=refcov,
                          crossn=z["cross_covariance_numerical"],crossp=z["cross_covariance_prior"],
                          all_seed_ids=train_ids|ref_ids)
    # Both charge samples are independent. Within each charge all model/input
    # correlations remain paired, including changes of the asymmetry Jacobian.
    p,m=full["WP"],full["WM"];assert p["meta"]["models"]==m["meta"]["models"]
    assert p["meta"]["pairs"]==m["meta"]["pairs"]
    assert p["meta"]["scale_names"]==m["meta"]["scale_names"]
    assert not p["all_seed_ids"]&m["all_seed_ids"]
    ix=np.arange(7,40);am=[];an=[];aprior=[];ader=[]
    def asym(pm,mm,pc,mc):
        total=pm+mm;assert (total>0).all()
        dplus=2*mm[0]/total[0]**2;dminus=-2*pm[0]/total[0]**2
        cov=dplus[:,None]*pc*dplus[None,:]+dminus[:,None]*mc*dminus[None,:]
        return (pm-mm)/total,cov,(dplus,dminus)
    for i,name in enumerate(p["meta"]["models"]):
        mean,n,der=asym(p["means"][i][:,ix],m["means"][i][:,ix],
                       p["numeric"][i][np.ix_(ix,ix)],m["numeric"][i][np.ix_(ix,ix)])
        _,pc,_=asym(p["means"][i][:,ix],m["means"][i][:,ix],
                   p["prior"][i][np.ix_(ix,ix)],m["prior"][i][np.ix_(ix,ix)])
        am.append(mean);an.append(n);aprior.append(pc);ader.append(der)
    am=np.asarray(am);an=np.asarray(an);aprior=np.asarray(aprior)
    ar,arc,_=asym(p["refmean"][:,ix],m["refmean"][:,ix],p["refcov"][np.ix_(ix,ix)],m["refcov"][np.ix_(ix,ix)])
    acn=[];acp=[];asympaired={}
    for k,(first,second) in enumerate(p["meta"]["pairs"]):
        i,j=p["meta"]["models"].index(first),p["meta"]["models"].index(second)
        cross=[]
        for kind in ("crossn","crossp"):
            c=np.zeros((33,33))
            for charge_index,ch in enumerate((p,m)):
                c+=ader[i][charge_index][:,None]*ch[kind][k][np.ix_(ix,ix)]*ader[j][charge_index][None,:]
            cross.append(c)
        dn=an[i]+an[j]-cross[0]-cross[0].T;dp=aprior[i]+aprior[j]-cross[1]-cross[1].T
        acn.append(dn);acp.append(dp);asympaired[first+" minus "+second]=summarize(am[i]-am[j],dn+dp)
    report["asymmetry"]=dict(labels=p["meta"]["labels"][7:40],reference=summarize(ar,arc),
        models={name:dict(prediction=summarize(am[i],an[i]+aprior[i]),
                         difference_from_reference=summarize(am[i]-ar,an[i]+aprior[i]+arc))
                for i,name in enumerate(p["meta"]["models"])},paired=asympaired)
    np.savez_compressed(a.output/"asymmetry.npz",means=am,covariance_numerical=an,covariance_prior=aprior,
        reference=ar,reference_covariance=arc,paired_covariance_numerical=np.asarray(acn),
        paired_covariance_prior=np.asarray(acp),metadata=np.array(json.dumps(dict(models=p["meta"]["models"],
        pairs=p["meta"]["pairs"],labels=p["meta"]["labels"][7:40]))))
    (a.output/"RESULTS.json").write_text(json.dumps(plain(report),indent=2,allow_nan=False)+"\n")
    (a.output/"VALIDATION_COMPLETE.json").write_text(json.dumps(dict(reference_independent=True,
        training_frozen=True,scale_order_checked=True,training_scale_order_evidence=scale_evidence,
        precision_passed=precision["all_precision_passed"],
        precision_misses_reported=not precision["all_precision_passed"],
        results_sha256=hashlib.sha256((a.output/"RESULTS.json").read_bytes()).hexdigest()),indent=2)+"\n")
    print("VALIDATION_COMPLETE",flush=True)


if __name__=="__main__":main()
