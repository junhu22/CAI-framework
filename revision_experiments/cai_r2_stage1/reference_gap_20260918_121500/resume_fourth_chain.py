import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.dont_write_bytecode=True
torch.set_num_threads(4)
from pyro.infer import MCMC,NUTS
ROOT=Path(r"C:\neutra-hmc-surrogate\surrogate_training"); HERE=Path(__file__).resolve().parent
S1=ROOT/"revision_experiments"/"cai_r2_stage1"/"run_20260917_094722_21d144f3db844a6b8a2e8167c5282dbe"; ROUT=S1/"routing_arrays"
sys.path.insert(0,str(ROOT/"revision_experiments")); import revlib as R
TARGETS=[f"ablation_obs{i}" for i in range(1,6)]+["sailboat_boundary_e1_075"]
SEED=45; NSAMP=WARMUP=500
def spec(name):
    a=np.load(ROUT/f"{name}_seed42_N8000.npz"); y=np.array(a["observation"],np.float32); fm=np.array(a["theta"],np.float32).mean(0)
    return (y,fm,R.load_d6(),.30) if name.startswith("ablation_") else (y,fm,R.load_broad_ext(),.55)
def Ufun(sur,y,sigma):
    yt=torch.tensor(y,dtype=torch.float32); K=len(y)
    def U(p):
        z=p["z"]; th=torch.exp(z); f=sur(th.unsqueeze(0)).squeeze(0)[:K]
        return .5*(((f-yt)/yt)**2).sum()/1e-3 + .5*(z**2).sum()/sigma**2
    return U
for name in TARGETS:
    final=HERE/f"{name}_reference_chains_4chain.npz"
    if final.exists():
        print(name,"SKIP_EXISTING",flush=True); continue
    y,fm,sur,sigma=spec(name); init=np.clip(fm*.8,.2,5.0)
    torch.manual_seed(SEED); ker=NUTS(potential_fn=Ufun(sur,y,sigma),target_accept_prob=.8,full_mass=False,jit_compile=False)
    mc=MCMC(ker,num_samples=NSAMP,warmup_steps=WARMUP,initial_params={"z":torch.log(torch.tensor(init,dtype=torch.float32))},disable_progbar=True)
    t=time.time(); mc.run(); dt=time.time()-t; ch=torch.exp(mc.get_samples()["z"]).detach().cpu().numpy()
    diag=mc.diagnostics(); div=diag.get("divergences",{}); ndiv=sum(len(v) for v in div.values()) if isinstance(div,dict) else len(div)
    np.savez_compressed(HERE/f"{name}_reference_chain4_seed45.npz",chain=ch,observation=y,init=init)
    old=np.load(HERE/f"{name}_reference_chains.npz")["chains"]; four=np.concatenate([old,ch[None,:,:]],0)
    np.savez_compressed(final,chains=four,observation=y)
    print(name,"DONE","time",round(dt,2),"div",ndiv,"mean",np.round(ch.mean(0),4).tolist(),flush=True)
print("RESUME_COMPLETE",flush=True)
