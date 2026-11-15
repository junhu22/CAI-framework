"""Compare fixed fast posteriors with matched exact NUTS references."""
import csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance, spearmanr

ROOT=Path(r"C:\neutra-hmc-surrogate\surrogate_training")
HERE=Path(__file__).resolve().parent
S1=ROOT/"revision_experiments"/"cai_r2_stage1"/"run_20260917_094722_21d144f3db844a6b8a2e8167c5282dbe"
ROUT=S1/"routing_arrays"
PS=ROOT/"revision_experiments"/"cai_r2_stage1"/"final_ext_verification_20260918_091227"/"PSIS_STAGE1_DETAIL.csv"
OBS=[f"ablation_obs{i}" for i in range(1,6)]+["sailboat_boundary_e1_075","sailboat_field_frozen"]
Ns=[2000,8000]
with PS.open(encoding="utf-8") as f: ps=list(csv.DictReader(f))
refsum=json.loads((HERE/"ALL_REFERENCE_SUMMARY.json").read_text())
def ref_samples(name):
    if name=="sailboat_field_frozen":
        a=np.load(ROOT/"revision_experiments"/"fig17_replay_samples.npz")
        return np.array(a["nuts_all"],float)
    return np.load(HERE/f"{name}_reference_chains.npz")["chains"].reshape(-1,6).astype(float)
def dist_metrics(fast,ref,idx):
    fm=fast.mean(0); rm=ref.mean(0)
    fq=np.quantile(fast,[.025,.5,.975],axis=0); rq=np.quantile(ref,[.025,.5,.975],axis=0)
    fw=fq[2]-fq[0]; rw=rq[2]-rq[0]
    w1=np.array([wasserstein_distance(fast[:,j],ref[:,j]) for j in range(6)])
    rs=ref.std(0,ddof=1); nw1=w1/np.maximum(rs,1e-12)
    return dict(mean_mae=float(np.abs(fm[idx]-rm[idx]).mean()),
                mean_w1=float(w1[idx].mean()),mean_norm_w1=float(nw1[idx].mean()),
                width_mae=float(np.abs(fw[idx]-rw[idx]).mean()),
                median_mae=float(np.abs(fq[1,idx]-rq[1,idx]).mean()),
                max_norm_w1=float(nw1[idx].max()))
rows=[]
for name in OBS:
    ref=ref_samples(name)
    for N in Ns:
        for seed in [42,43,44,45,46]:
            a=np.load(ROUT/f"{name}_seed{seed}_N{N}.npz"); fast=np.array(a["theta"],float)
            active=dist_metrics(fast,ref,np.array([0,1,2]))
            inactive=dist_metrics(fast,ref,np.array([3,4,5]))
            pr=next(r for r in ps if r["observation"]==name and int(r["seed"])==seed and int(r["N"])==N)
            rows.append(dict(observation=name,N=N,seed=seed,raw_ess=float(pr["raw_ess_ratio"]),
                             psis_ess=float(pr["psis_ess_ratio"]),pareto_k=float(pr["pareto_k"]),
                             raw_route=pr["route_raw_tau05"],psis_route=pr["route_psis_tau05"],
                             **{f"active_{k}":v for k,v in active.items()},
                             **{f"inactive_{k}":v for k,v in inactive.items()}))
with (HERE/"ROUTING_GAP_DETAIL.csv").open("x",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summary=[]
for name in OBS:
  for N in Ns:
    rs=[r for r in rows if r["observation"]==name and r["N"]==N]
    rr=np.array([r["raw_ess"] for r in rs]); pe=np.array([r["psis_ess"] for r in rs]); kk=np.array([r["pareto_k"] for r in rs])
    mm=np.array([r["active_mean_mae"] for r in rs]); w1=np.array([r["active_mean_w1"] for r in rs]); nw=np.array([r["active_mean_norm_w1"] for r in rs]); wd=np.array([r["active_width_mae"] for r in rs])
    s=refsum[name]
    summary.append(dict(observation=name,N=N,raw_ess_mean=float(rr.mean()),raw_ess_min=float(rr.min()),raw_ess_max=float(rr.max()),
      psis_ess_mean=float(pe.mean()),pareto_k_mean=float(kk.mean()),pareto_k_max=float(kk.max()),
      raw_correction_n=sum(r["raw_route"]=="CORRECTION" for r in rs),psis_correction_n=sum(r["psis_route"]=="CORRECTION" for r in rs),
      active_mean_mae_mean=float(mm.mean()),active_mean_mae_max=float(mm.max()),
      active_w1_mean=float(w1.mean()),active_norm_w1_mean=float(nw.mean()),active_width_mae_mean=float(wd.mean()),
      reference_ess_min=float(s["ess_min"]),reference_rhat_max=float(s["rhat_max"])))
with (HERE/"ROUTING_GAP_SUMMARY.csv").open("x",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
# Descriptive correlations use one point per target at the operational N=2000.
op=[r for r in summary if r["N"]==2000]
raw=np.array([r["raw_ess_mean"] for r in op]); pse=np.array([r["psis_ess_mean"] for r in op])
gap=np.array([r["active_mean_mae_mean"] for r in op]); wgap=np.array([r["active_norm_w1_mean"] for r in op])
corr={"N":2000,"n_targets":len(op),
      "spearman_rawESS_vs_active_meanMAE":list(map(float,spearmanr(raw,gap))),
      "spearman_PSIS_ESS_vs_active_meanMAE":list(map(float,spearmanr(pse,gap))),
      "spearman_rawESS_vs_active_normW1":list(map(float,spearmanr(raw,wgap))),
      "spearman_PSIS_ESS_vs_active_normW1":list(map(float,spearmanr(pse,wgap)))}
(HERE/"ROUTING_GAP_CORRELATIONS.json").write_text(json.dumps(corr,indent=2),encoding="utf-8")
print("SUMMARY N=2000")
for r in op: print(r)
print("CORRELATIONS",corr)
