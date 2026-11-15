import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(6)
ROOT=Path(r'C:\neutra-hmc-surrogate\surrogate_training')
HERE=Path(r'C:\neutra-hmc-surrogate\surrogate_training\revision_experiments\cai_r2_stage1\final_ext_verification_20260918_091227')
OUT=HERE/'ablation_multiseed'; OUT.mkdir(exist_ok=True)
CKPT=Path(r'C:\nf-pmcmc\experiments\results\ablation_checkpoint.pt')
sys.path.insert(0,str(ROOT/'experiments'))
from cai_lib import ConditionalRealNVP
blob=torch.load(CKPT,map_location='cpu',weights_only=False)
t_all=blob['t_all'].float(); y_all=blob['y_all'].float()
t_val,y_val=t_all[100000:105000],y_all[100000:105000]
CONFIGS={'N5000_L8':(5000,8),'N10000_L8':(10000,8),'N25000_L8':(25000,8),'N50000_L8':(50000,8),'N100000_L8':(100000,8),'N50000_L4':(50000,4),'N50000_L6':(50000,6),'N50000_L12':(50000,12)}
SEEDS=[101,202,303]; FAST_SEEDS=[42,43,44,45,46]
refs=[]; obs=[]
for i in range(1,6):
    a=np.load(HERE/f'REFERENCE_ablation_obs{i}.npz')
    refs.append(a['ref_mean'].astype(np.float32)); obs.append(a['observation'].astype(np.float32))
refs=np.stack(refs); obs=np.stack(obs)
assert t_all.shape==(105000,6) and y_all.shape==(105000,20)
def train(flow,t_tr,y_tr,seed,max_epochs=300):
    torch.manual_seed(seed)
    opt=torch.optim.Adam(flow.parameters(),lr=1e-3,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max_epochs)
    ds=torch.utils.data.TensorDataset(t_tr,y_tr)
    gen=torch.Generator().manual_seed(seed)
    ld=torch.utils.data.DataLoader(ds,batch_size=512,shuffle=True,drop_last=True,generator=gen)
    best=float('inf'); best_st=None; noimp=0; best_ep=0; steps=0
    t0=time.time()
    for ep in range(1,max_epochs+1):
        flow.train()
        for tb,yb in ld:
            z,ldet=flow(tb,yb); loss=-((-0.5*(z**2).sum(-1))+ldet).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(flow.parameters(),5.0); opt.step(); steps+=1
        sch.step(); flow.eval()
        with torch.no_grad():
            zv,ldv=flow(t_val,y_val); vl=-((-0.5*(zv**2).sum(-1))+ldv).mean().item()
        if vl<best:
            best=vl; best_st={k:v.detach().cpu().clone() for k,v in flow.state_dict().items()}; noimp=0; best_ep=ep
        else:
            noimp+=1
        if noimp>=30: break
    flow.load_state_dict(best_st); flow.eval()
    return flow,best,best_ep,ep,steps,time.time()-t0
def evaluate(flow):
    per_fast=[]
    for fs in FAST_SEEDS:
        torch.manual_seed(fs)
        errs_all=[]; errs_act=[]
        for y,ref in zip(obs,refs):
            yt=torch.tensor(y,dtype=torch.float32)
            pm=flow.sample(yt,2000).mean(0).numpy()
            errs_all.append(float(np.mean(np.abs(pm-ref))))
            errs_act.append(float(np.mean(np.abs(pm[:3]-ref[:3]))))
        per_fast.append((fs,float(np.mean(errs_all)),float(np.mean(errs_act))))
    arr=np.array([[x[1],x[2]] for x in per_fast])
    return per_fast,float(arr[:,0].mean()),float(arr[:,1].mean()),float(arr[:,0].std(ddof=1)),float(arr[:,1].std(ddof=1))
def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
ap=argparse.ArgumentParser(); ap.add_argument('--configs',nargs='*',default=list(CONFIGS)); ap.add_argument('--seeds',nargs='*',type=int,default=SEEDS); args=ap.parse_args()
manifest={'checkpoint':str(CKPT),'checkpoint_sha256':sha(CKPT),'configs':args.configs,'seeds':args.seeds,'fast_seeds':FAST_SEEDS,'reference_source':str(HERE/'REFERENCE_GAP_SUMMARY.csv'),'metric':'MAE vs matched NUTS reference; all6 plus active e1-e3','train_rule':'same frozen nested bank, same validation, min validation loss, patience30, no test selection'}
(OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
rows=[]
if (OUT/'RESULTS.csv').exists():
    rows=list(csv.DictReader((OUT/'RESULTS.csv').open(encoding='utf-8')))
done={(r['config'],int(r['training_seed'])) for r in rows}
for cid in args.configs:
    n,layers=CONFIGS[cid]
    for seed in args.seeds:
        if (cid,seed) in done:
            print('SKIP',cid,seed,flush=True); continue
        print('RUN',cid,'seed',seed,flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        flow=ConditionalRealNVP(6,20,n_layers=layers,hidden_dim=128)
        flow,best,best_ep,last_ep,steps,dt=train(flow,t_all[:n],y_all[:n],seed)
        perf,mae6,mae3,sd6,sd3=evaluate(flow)
        ck=OUT/f'{cid}_seed{seed}.pt'; torch.save(flow.state_dict(),ck)
        row=dict(config=cid,n_train=n,n_layers=layers,training_seed=seed,val_loss=best,best_epoch=best_ep,last_epoch=last_ep,optimizer_steps=steps,train_time_s=dt,mae_ref_all6=mae6,mae_ref_active=mae3,fast_seed_sd_all6=sd6,fast_seed_sd_active=sd3,checkpoint=str(ck),fast_seed_results=json.dumps(perf))
        rows.append(row)
        with (OUT/'RESULTS.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(row)); w.writeheader(); w.writerows(rows)
        print('DONE',cid,seed,'val',best,'mae6',mae6,'mae3',mae3,'time',dt,flush=True)
print('ALL_REQUESTED_DONE',len(rows),flush=True)
