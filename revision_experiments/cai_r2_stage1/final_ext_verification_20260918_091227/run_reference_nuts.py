import csv, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from pyro.infer import MCMC, NUTS
torch.set_num_threads(6)
ROOT=Path(r'C:\neutra-hmc-surrogate\surrogate_training')
HERE=Path(r'C:\neutra-hmc-surrogate\surrogate_training\revision_experiments\cai_r2_stage1\final_ext_verification_20260918_091227')
S1=Path(r'C:\neutra-hmc-surrogate\surrogate_training\revision_experiments\cai_r2_stage1\run_20260917_094722_21d144f3db844a6b8a2e8167c5282dbe')
sys.path.insert(0,str(ROOT/'revision_experiments'))
import revlib as R
from cai_lib import load_surrogate, effective_sample_size
D6_SUR=ROOT/'surrogate_models'/'member_0.pt'
EXT_SUR=ROOT/'surrogate_models_broad_ext'/'member_0.pt'
sur_d6=load_surrogate(D6_SUR); sur_ext=load_surrogate(EXT_SUR)
NSAMP=600; NWARM=500; SEEDS=[701,702,703]
def split_rhat(chains):
    return np.asarray(R.split_rhat(chains),float)
def run_chain(sur,y,sigma,init,seed):
    yt=torch.tensor(y,dtype=torch.float32); K=len(y)
    def U(p):
        z=p['z']; f=sur(torch.exp(z).unsqueeze(0)).squeeze(0)[:K]
        return 0.5*(((f-yt)/yt)**2).sum()/1e-3 + 0.5*(z**2).sum()/(sigma**2)
    torch.manual_seed(seed)
    mc=MCMC(NUTS(potential_fn=U,target_accept_prob=0.8),num_samples=NSAMP,warmup_steps=NWARM,
            initial_params={'z':torch.log(torch.tensor(init,dtype=torch.float32))},disable_progbar=True)
    t=time.time(); mc.run(); dt=time.time()-t
    th=torch.exp(mc.get_samples()['z']).detach().cpu()
    diag=mc.diagnostics()
    return th.numpy(),dt,diag
from scipy.stats import wasserstein_distance
targets=[]
for name in [f'ablation_obs{i}' for i in range(1,6)]:
    p=S1/'routing_arrays'/f'{name}_seed42_N8000.npz'; a=np.load(p)
    targets.append(dict(name=name,sur='d6',y=a['observation'].astype(float),sigma=0.30))
for name,sigma in [('sailboat_field_frozen',0.30),('sailboat_boundary_e1_075',0.55)]:
    p=S1/'routing_arrays'/f'{name}_seed42_N8000.npz'; a=np.load(p)
    targets.append(dict(name=name,sur='ext',y=a['observation'].astype(float),sigma=sigma))
rows=[]; all_arrays={}
for ti,tg in enumerate(targets):
    name=tg['name']; sur=sur_d6 if tg['sur']=='d6' else sur_ext; y=tg['y']; sigma=tg['sigma']
    fast=[]
    for s in [42,43,44,45,46]:
        fast.append(np.load(S1/'routing_arrays'/f'{name}_seed{s}_N8000.npz')['theta'])
    fast=np.concatenate(fast,axis=0)
    init_fast=np.clip(np.nanmean(fast,axis=0),0.15,5.0)
    inits=[init_fast,np.ones(6),np.exp(np.linspace(-0.20,0.20,6))]
    chains=[]; times=[]; diags=[]
    print('TARGET',name,'K',len(y),'sigma',sigma,'fast_mean',init_fast.tolist(),flush=True)
    for ci,(seed,init) in enumerate(zip(SEEDS,inits)):
        ch,dt,diag=run_chain(sur,y,sigma,init,seed); chains.append(ch); times.append(dt); diags.append(diag)
        ess=effective_sample_size(torch.tensor(ch,dtype=torch.float32)).numpy()
        acc=diag.get('acceptance rate',{}).get('z',None)
        div=diag.get('divergences',{}).get('z',[])
        print(' CHAIN',ci+1,'time',round(dt,2),'essmin',float(np.nanmin(ess)),'acc',acc,'div',len(div) if hasattr(div,'__len__') else div,flush=True)
    chains=np.stack(chains); ref=chains.reshape(-1,6)
    rhat=split_rhat([c for c in chains])
    essmins=[float(np.min(effective_sample_size(torch.tensor(c,dtype=torch.float32)).numpy())) for c in chains]
    fq=np.quantile(fast,[.025,.25,.5,.75,.975],axis=0); rq=np.quantile(ref,[.025,.25,.5,.75,.975],axis=0)
    fm=fast.mean(0); rm=ref.mean(0)
    w1=np.array([wasserstein_distance(fast[:,j],ref[:,j]) for j in range(6)])
    widthf=fq[4]-fq[0]; widthr=rq[4]-rq[0]
    rows.append(dict(target=name,K=len(y),sigma_log=sigma,n_fast=len(fast),n_ref=len(ref),
                     mean_mae_active=float(np.mean(np.abs(fm[:3]-rm[:3]))),
                     mean_mae_inactive=float(np.mean(np.abs(fm[3:]-rm[3:]))),
                     mean_mae_all6=float(np.mean(np.abs(fm-rm))),
                     w1_active_mean=float(w1[:3].mean()),w1_active_max=float(w1[:3].max()),
                     q95_width_ratio_active_mean=float(np.mean(widthf[:3]/widthr[:3])),
                     q95_width_ratio_active_min=float(np.min(widthf[:3]/widthr[:3])),
                     q95_width_ratio_active_max=float(np.max(widthf[:3]/widthr[:3])),
                     rhat_max=float(np.nanmax(rhat)),rhat_active_max=float(np.nanmax(rhat[:3])),
                     essmin_chain_min=float(np.min(essmins)),essmin_chain_mean=float(np.mean(essmins)),
                     runtime_s=float(np.sum(times))))
    all_arrays[name]=dict(chains=chains,fast=fast,fast_quantiles=fq,ref_quantiles=rq,fast_mean=fm,ref_mean=rm,w1=w1,rhat=rhat)
    np.savez_compressed(HERE/f'REFERENCE_{name}.npz',chains=chains,fast=fast,fast_quantiles=fq,ref_quantiles=rq,fast_mean=fm,ref_mean=rm,w1=w1,rhat=rhat,observation=y)
    print(' SUMMARY',rows[-1],flush=True)
with (HERE/'REFERENCE_GAP_SUMMARY.csv').open('x',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
json.dump({'settings':{'samples_per_chain':NSAMP,'warmup':NWARM,'seeds':SEEDS,'sampler':'Pyro NUTS in log-theta','target_accept':0.8},'rows':rows},
          open(HERE/'REFERENCE_RUN_SUMMARY.json','x'),indent=2)
print('DONE',json.dumps(rows),flush=True)
