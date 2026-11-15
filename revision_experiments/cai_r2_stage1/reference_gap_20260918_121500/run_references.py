"""Bounded matched-reference runs for Reviewer 2.1 routing-vs-gap validation."""
import hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
sys.dont_write_bytecode=True
torch.set_num_threads(4)
from pyro.infer import MCMC, NUTS
from pyro.ops.stats import effective_sample_size, split_gelman_rubin

ROOT=Path(r"C:\neutra-hmc-surrogate\surrogate_training")
HERE=Path(__file__).resolve().parent
S1=ROOT/"revision_experiments"/"cai_r2_stage1"/"run_20260917_094722_21d144f3db844a6b8a2e8167c5282dbe"
ROUT=S1/"routing_arrays"
sys.path.insert(0,str(ROOT/"revision_experiments"))
import revlib as R

TARGETS=[f"ablation_obs{i}" for i in range(1,6)]+["sailboat_boundary_e1_075"]
SEEDS=[42,43,44]
NSAMP=500
WARMUP=500
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git_state():
    return subprocess.run(["git","--no-optional-locks","-C",str(ROOT),"status","--short","--untracked-files=no"],
                          capture_output=True,text=True,check=True).stdout
def load_spec(name):
    a=np.load(ROUT/f"{name}_seed42_N8000.npz")
    y=np.array(a["observation"],dtype=np.float32)
    fm=np.array(a["theta"],dtype=np.float32).mean(0)
    if name.startswith("ablation_"):
        model_path=ROOT/"surrogate_models"/"member_0.pt"
        sur=R.load_d6(); sigma=0.30
    else:
        model_path=ROOT/"surrogate_models_broad_ext"/"member_0.pt"
        sur=R.load_broad_ext(); sigma=0.55
    return y,fm,sur,sigma,model_path

def potential(sur,y,sigma):
    yt=torch.tensor(y,dtype=torch.float32); K=len(y)
    def U(params):
        z=params["z"]; theta=torch.exp(z)
        f=sur(theta.unsqueeze(0)).squeeze(0)[:K]
        rel=(f-yt)/yt
        return 0.5*(rel**2).sum()/1e-3 + 0.5*(z**2).sum()/sigma**2
    return U
def count_divergences(diag):
    d=diag.get("divergences",{}) if isinstance(diag,dict) else {}
    if isinstance(d,dict):
        return int(sum(len(v) if hasattr(v,"__len__") else int(bool(v)) for v in d.values()))
    if hasattr(d,"__len__"): return int(len(d))
    return int(bool(d))

def run_chain(U,init,seed):
    torch.manual_seed(seed)
    kernel=NUTS(potential_fn=U,target_accept_prob=0.8,full_mass=False,jit_compile=False)
    mc=MCMC(kernel,num_samples=NSAMP,warmup_steps=WARMUP,
            initial_params={"z":torch.log(torch.tensor(init,dtype=torch.float32))},
            disable_progbar=True)
    t=time.time(); mc.run(); dt=time.time()-t
    th=torch.exp(mc.get_samples()["z"]).detach().cpu().numpy()
    diag=mc.diagnostics()
    return th,dict(seed=seed,time_s=dt,step_size=float(kernel.step_size),
                   mean_accept_prob=float(getattr(kernel,"_mean_accept_prob",float("nan"))),
                   divergences=count_divergences(diag))
initial_git=git_state()
manifest={"targets":TARGETS,"seeds":SEEDS,"num_samples":NSAMP,"warmup":WARMUP,
          "target_accept_prob":0.8,"parameterisation":"z=log(theta)",
          "target":"surrogate likelihood + N(0,sigma^2) prior in z",
          "git_tracked_before":initial_git,
          "field_reference":"reuse existing fig17_replay_samples.npz; no field rerun",
          "boundaries":["no training","no manuscript edit","no new FE","only six new NUTS targets"]}
(HERE/"REFERENCE_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

all_summary={}
for name in TARGETS:
    y,fm,sur,sigma,model_path=load_spec(name)
    U=potential(sur,y,sigma)
    inits=[np.ones(6),np.clip(fm,0.2,5.0),np.clip(fm*1.2,0.2,5.0)]
    chains=[]; cdiag=[]
    print("TARGET",name,"K",len(y),"sigma",sigma,"fastmean",fm.tolist(),flush=True)
    for ci,(init,seed) in enumerate(zip(inits,SEEDS)):
        th,d=run_chain(U,init,seed)
        chains.append(th); cdiag.append(d)
        print(" chain",ci,"seed",seed,"time",round(d["time_s"],2),
              "step",round(d["step_size"],5),"div",d["divergences"],
              "mean",np.round(th.mean(0),4).tolist(),flush=True)
    arr=np.stack(chains)
    ten=torch.tensor(arr,dtype=torch.float32)
    ess=effective_sample_size(ten,chain_dim=0,sample_dim=1).cpu().numpy()
    rhat=split_gelman_rubin(ten,chain_dim=0,sample_dim=1).cpu().numpy()
    np.savez_compressed(HERE/f"{name}_reference_chains.npz",chains=arr,observation=y,
                        fastmean_seed42_N8000=fm,inits=np.stack(inits))
    s={"target":name,"K":len(y),"sigma_log":sigma,"model_path":str(model_path),
       "model_sha256":sha(model_path),"chain_diagnostics":cdiag,
       "ess_per_dim":ess.tolist(),"ess_min":float(np.nanmin(ess)),
       "rhat_per_dim":rhat.tolist(),"rhat_max":float(np.nanmax(rhat)),
       "reference_mean":arr.reshape(-1,6).mean(0).tolist(),
       "reference_q025":np.quantile(arr.reshape(-1,6),.025,axis=0).tolist(),
       "reference_q975":np.quantile(arr.reshape(-1,6),.975,axis=0).tolist()}
    all_summary[name]=s
    (HERE/f"{name}_reference_summary.json").write_text(json.dumps(s,indent=2),encoding="utf-8")
    print(" reference ESSmin",round(s["ess_min"],1),"Rhatmax",round(s["rhat_max"],5),flush=True)
field_npz=ROOT/"revision_experiments"/"fig17_replay_samples.npz"
field_res=ROOT/"revision_experiments"/"fig17_replay_results.json"
fa=np.load(field_npz); fchains=np.array(fa["nuts_all"]).reshape(3,500,6)
ften=torch.tensor(fchains,dtype=torch.float32)
fess=effective_sample_size(ften,chain_dim=0,sample_dim=1).cpu().numpy()
frhat=split_gelman_rubin(ften,chain_dim=0,sample_dim=1).cpu().numpy()
fr=json.loads(field_res.read_text())
all_summary["sailboat_field_frozen"]={"target":"sailboat_field_frozen","K":6,"sigma_log":0.30,
    "reference_source":str(field_npz),"source_sha256":sha(field_npz),
    "ess_per_dim":fess.tolist(),"ess_min":float(np.nanmin(fess)),
    "rhat_per_dim":frhat.tolist(),"rhat_max":float(np.nanmax(frhat)),
    "reference_mean":fchains.reshape(-1,6).mean(0).tolist(),
    "reported_replay_gate":fr["gate"]}
(HERE/"ALL_REFERENCE_SUMMARY.json").write_text(json.dumps(all_summary,indent=2),encoding="utf-8")
assert git_state()==initial_git,"tracked files changed during reference runs"
print("ALL_REFERENCE_RUNS_COMPLETE",flush=True)
