import csv,json
from pathlib import Path
import numpy as np, arviz as az
from scipy.stats import wasserstein_distance,spearmanr
ROOT=Path(r"C:\neutra-hmc-surrogate\surrogate_training"); HERE=Path(r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments\cai_r2_stage1\reference_gap_20260918_121500")
S1=ROOT/"revision_experiments"/"cai_r2_stage1"/"run_20260917_094722_21d144f3db844a6b8a2e8167c5282dbe"; ROUT=S1/"routing_arrays"
PS=ROOT/"revision_experiments"/"cai_r2_stage1"/"final_ext_verification_20260918_091227"/"PSIS_STAGE1_DETAIL.csv"
NEW=[f"ablation_obs{i}" for i in range(1,6)]+["sailboat_boundary_e1_075"]; OBS=NEW+["sailboat_field_frozen"]
with PS.open(encoding="utf-8") as f: ps=list(csv.DictReader(f))
refs={}; diagnostics={}
for n in OBS:
    if n=="sailboat_field_frozen":
        a=np.load(ROOT/"revision_experiments"/"fig17_replay_samples.npz"); ch=a["nuts_all"].reshape(3,500,6)
        planned=False
    else:
        ch=np.load(HERE/f"{n}_reference_chains_4chain.npz")["chains"]; planned=True
    refs[n]=ch.reshape(-1,6).astype(float)
    idata=az.from_dict(posterior={"theta":ch})
    rh=np.asarray(az.rhat(idata,var_names=["theta"],method="rank")["theta"]); eb=np.asarray(az.ess(idata,var_names=["theta"],method="bulk")["theta"]); et=np.asarray(az.ess(idata,var_names=["theta"],method="tail")["theta"])
    diagnostics[n]={"n_chains":int(ch.shape[0]),"draws_per_chain":int(ch.shape[1]),"rank_rhat_max":float(rh.max()),"bulk_ess_min":float(eb.min()),"tail_ess_min":float(et.min()),"predeclared_new_reference_criteria_applicable":planned,"criteria_pass":bool((not planned) or (ch.shape[0]>=4 and rh.max()<=1.01 and eb.min()>=400 and et.min()>=400))}
    print(n,diagnostics[n],flush=True)
(HERE/"FINAL_REFERENCE_DIAGNOSTICS.json").write_text(json.dumps({"arviz":az.__version__,"targets":diagnostics},indent=2),encoding="utf-8")
def metrics(fast,ref,idx):
    fm=fast.mean(0); rm=ref.mean(0); fq=np.quantile(fast,[.025,.5,.975],axis=0); rq=np.quantile(ref,[.025,.5,.975],axis=0)
    w1=np.array([wasserstein_distance(fast[:,j],ref[:,j]) for j in range(6)]); nw1=w1/np.maximum(ref.std(0,ddof=1),1e-12)
    return {"mean_mae":float(np.abs(fm[idx]-rm[idx]).mean()),"norm_w1":float(nw1[idx].mean()),"width_mae":float(np.abs((fq[2]-fq[0])[idx]-(rq[2]-rq[0])[idx]).mean())}
rows=[]
for n in OBS:
    for N in [2000,8000]:
        for seed in [42,43,44,45,46]:
            a=np.load(ROUT/f"{n}_seed{seed}_N{N}.npz"); fast=np.asarray(a["theta"],float)
            pr=next(r for r in ps if r["observation"]==n and int(r["N"])==N and int(r["seed"])==seed)
            m=metrics(fast,refs[n],np.array([0,1,2]))
            rows.append({"observation":n,"N":N,"seed":seed,"raw_ess":float(pr["raw_ess_ratio"]),"psis_ess":float(pr["psis_ess_ratio"]),"pareto_k":float(pr["pareto_k"]),"raw_route":pr["route_raw_tau05"],"psis_route":pr["route_psis_tau05"],"active_mean_mae":m["mean_mae"],"active_norm_w1":m["norm_w1"],"active_width_mae":m["width_mae"]})
with (HERE/"ROUTING_GAP_DETAIL_4CHAIN.csv").open("x",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
sumrows=[]
for n in OBS:
    for N in [2000,8000]:
        z=[r for r in rows if r["observation"]==n and r["N"]==N]
        sumrows.append({"observation":n,"N":N,"raw_ess_mean":float(np.mean([r["raw_ess"] for r in z])),"psis_ess_mean":float(np.mean([r["psis_ess"] for r in z])),"pareto_k_mean":float(np.mean([r["pareto_k"] for r in z])),"raw_correction_n":sum(r["raw_route"]=="CORRECTION" for r in z),"psis_correction_n":sum(r["psis_route"]=="CORRECTION" for r in z),"active_mean_mae_mean":float(np.mean([r["active_mean_mae"] for r in z])),"active_mean_mae_max":float(np.max([r["active_mean_mae"] for r in z])),"active_norm_w1_mean":float(np.mean([r["active_norm_w1"] for r in z])),"active_width_mae_mean":float(np.mean([r["active_width_mae"] for r in z])),"reference_rank_rhat_max":diagnostics[n]["rank_rhat_max"],"reference_bulk_ess_min":diagnostics[n]["bulk_ess_min"],"reference_tail_ess_min":diagnostics[n]["tail_ess_min"]})
with (HERE/"ROUTING_GAP_SUMMARY_4CHAIN.csv").open("x",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(sumrows[0])); w.writeheader(); w.writerows(sumrows)
op=[r for r in sumrows if r["N"]==2000]; raw=np.array([r["raw_ess_mean"] for r in op]); pse=np.array([r["psis_ess_mean"] for r in op]); gap=np.array([r["active_mean_mae_mean"] for r in op]); wg=np.array([r["active_norm_w1_mean"] for r in op])
corr={"raw_vs_mean_mae":list(map(float,spearmanr(raw,gap))),"psis_vs_mean_mae":list(map(float,spearmanr(pse,gap))),"raw_vs_norm_w1":list(map(float,spearmanr(raw,wg))),"psis_vs_norm_w1":list(map(float,spearmanr(pse,wg)))}
(HERE/"ROUTING_GAP_CORRELATIONS_4CHAIN.json").write_text(json.dumps(corr,indent=2),encoding="utf-8")
print("N2000"); [print(r) for r in op]; print("CORR",corr)
