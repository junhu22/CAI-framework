"""R3-T4: E3 damage grid with the FINAL domain-extended surrogate+CNF (lean: 12 reps, 120 null)."""
import sys, json
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(6)
sys.path.insert(0, r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments")
import revlib as R
HERE = Path(__file__).parent
torch.manual_seed(42); np.random.seed(42)
sur = R.load_broad_ext(); cnf = R.load_broad_ext_cnf()
POS = {"e1": 0, "e3": 2, "e5": 4}; SEV = [0.10, 0.25, 0.50]; KAP = [0.01, 0.03, 0.05]
NREP, NNULL, NFAST = 12, 120, 1000
def surro(t):
    with torch.no_grad(): return sur(torch.tensor(t, dtype=torch.float32).reshape(1,-1)).squeeze(0).numpy()
def fmean(y): return cnf.sample(torch.tensor(y, dtype=torch.float32), NFAST).numpy().mean(0)
def auc(p, n):
    a=np.concatenate([p,n]); r=a.argsort().argsort().astype(float)+1
    return (r[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n))
null={k:[] for k in KAP}
for k in KAP:
    for _ in range(NNULL):
        th=np.clip(np.exp(0.55*np.random.randn(6)),0.5,3.5); y=surro(th); y=y+k*np.abs(y)*np.random.randn(20)
        null[k].append(fmean(y))
    null[k]=np.array(null[k])
cells={}
for pn,pi in POS.items():
    for s in SEV:
        for k in KAP:
            dm,loc=[],[]
            for _ in range(NREP):
                th=np.ones(6); th[pi]=1-s; y=surro(th); y=y+k*np.abs(y)*np.random.randn(20)
                pm=fmean(y); dm.append(pm); loc.append(abs(pm[pi]-th[pi])/th[pi])
            dm=np.array(dm)
            cells[(pn,s,k)]=dict(auc=auc(-dm[:,pi],-null[k][:,pi]), loc_err=float(np.mean(loc)))
summary={f"sev{int(s*100)}":{"auc":float(np.mean([cells[(p,s,k)]['auc'] for p in POS for k in KAP])),
                             "loc_err_pct":float(np.mean([cells[(p,s,k)]['loc_err'] for p in POS for k in KAP])*100)} for s in SEV}
json.dump({"summary":summary,"cells":{f"{p}|{s}|{k}":cells[(p,s,k)] for (p,s,k) in cells},
           "note":"ext models; manual reference AUC 0.62/0.69/0.77 was round-1 corrupted surrogate"},
          open(HERE/"e3_ext_results.json","w"), indent=2)
for s in SEV: print(f"sev {int(s*100)}%: AUC={summary[f'sev{int(s*100)}']['auc']:.3f} loc_err={summary[f'sev{int(s*100)}']['loc_err_pct']:.1f}%", flush=True)
print("Wrote e3_ext_results.json", flush=True)
