"""R3-T1 fix: Case-B points from the NARROW-prior (sigma=0.30) field posterior (e1~2.2,
within the well-surrogated domain) -> validate the gate passes with the recommended config."""
import sys, csv, json
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(6)
sys.path.insert(0, r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments")
import revlib as R
from pyro.infer import MCMC, NUTS
HERE = Path(__file__).parent
torch.manual_seed(42); np.random.seed(42)
sur = R.load_broad_ext(); ens = R.load_broad_ext_ensemble()
yt = torch.tensor(R.FIELD_MEASURED)
def pot(p):
    z=p["z"]; f=sur(torch.exp(z).unsqueeze(0)).squeeze(0)[:6]
    return -(-0.5*(((f-yt)/yt)**2).sum()/1e-3 - 0.5*(z**2).sum()/0.30**2)  # narrow sigma=0.30
mc=MCMC(NUTS(potential_fn=pot,target_accept_prob=0.8),num_samples=1500,warmup_steps=500,
        initial_params={"z":torch.log(torch.tensor(R.FIELD_FASTMEAN,dtype=torch.float32))},disable_progbar=True)
mc.run(); th=torch.exp(mc.get_samples()["z"]).numpy()
print(f"narrow-prior posterior: e1 mean={th[:,0].mean():.2f} max={th[:,0].max():.2f} (domain 3.5)",flush=True)
idx=np.random.choice(len(th),47,replace=False)
pts=[np.clip(th[i],0.5,3.5) for i in idx]  # clamp to training domain
pts.append(np.clip(th.mean(0),0.5,3.5))
lp=R.make_log_post(sur,R.FIELD_MEASURED,"broad"); lps=lp.forward(torch.tensor(th,dtype=torch.float32)).detach().numpy()
pts.append(np.clip(th[int(lps.argmax())],0.5,3.5)); pts.append(np.clip(th[int(lps.argmax())]+0.03*np.random.randn(6),0.5,3.5))
kinds=["random"]*47+["post_mean","MAP","near_MAP"]
tt=torch.tensor(np.array(pts),dtype=torch.float32)
with torch.no_grad(): fh=np.mean([m(tt).numpy() for m in ens],axis=0)
with open(HERE/"samples_caseB_narrow.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow([f"e{i}" for i in range(1,7)])
    for p in pts: w.writerow([f"{v:.10f}" for v in p])
json.dump([dict(case="Bnarrow",local_idx=i,kind=kinds[i],theta=[float(v) for v in pts[i]],
                fhat=[float(v) for v in fh[i]],y_obs=list(R.FIELD_MEASURED),K=6) for i in range(len(pts))],
          open(HERE/"points_metadata_narrow.json","w"),indent=1)
print("wrote samples_caseB_narrow.csv",flush=True)
