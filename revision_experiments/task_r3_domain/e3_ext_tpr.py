"""R3-T4b: seed-identical REPLAY of e3_ext.py that additionally records the detection
probability at a 5% false-positive rate (TPR@5%FPR).

This is NOT a new experiment: the RNG consumption order is byte-for-byte the same as
e3_ext.py (same seeds, same draw sequence, same models), so the AUCs must reproduce the
frozen values in e3_ext_results.json exactly.  The script asserts that before writing
anything; if the assertion fails the replay is discarded and no number is planted.

Detection score s = -(posterior mean of the damaged parameter): larger s = more damage.
Threshold = 95th percentile of the null scores (=> FPR 5%); TPR = fraction of damaged
realisations above it.
"""
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
def tpr_at_fpr(p, n, fpr=0.05):
    thr = np.quantile(n, 1.0 - fpr)          # 5% of nulls exceed thr
    return float((p > thr).mean())

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
            cells[(pn,s,k)]=dict(auc=auc(-dm[:,pi],-null[k][:,pi]),
                                 tpr05=tpr_at_fpr(-dm[:,pi],-null[k][:,pi]),
                                 loc_err=float(np.mean(loc)))

# --- gate: AUCs must match the frozen run exactly -----------------------------------
frozen = json.load(open(HERE/"e3_ext_results.json"))
bad = []
for (p,s,k), v in cells.items():
    ref = frozen["cells"][f"{p}|{s}|{k}"]["auc"]
    if abs(v["auc"] - ref) > 1e-12: bad.append((p,s,k,v["auc"],ref))
if bad:
    print("REPLAY MISMATCH -- discarding, no numbers planted:")
    for b in bad[:5]: print("  ", b)
    sys.exit(1)
print("Replay reproduces all 27 frozen AUCs exactly.", flush=True)

summary={f"sev{int(s*100)}":{
    "auc":float(np.mean([cells[(p,s,k)]['auc'] for p in POS for k in KAP])),
    "tpr05":float(np.mean([cells[(p,s,k)]['tpr05'] for p in POS for k in KAP])),
    "loc_err_pct":float(np.mean([cells[(p,s,k)]['loc_err'] for p in POS for k in KAP])*100)} for s in SEV}
by_pos={p:{f"sev{int(s*100)}":{"auc":float(np.mean([cells[(p,s,k)]['auc'] for k in KAP])),
                               "tpr05":float(np.mean([cells[(p,s,k)]['tpr05'] for k in KAP]))}
           for s in SEV} for p in POS}
json.dump({"summary":summary,"by_position":by_pos,
           "cells":{f"{p}|{s}|{k}":cells[(p,s,k)] for (p,s,k) in cells},
           "note":"seed-identical replay of e3_ext.py; AUCs verified identical to e3_ext_results.json"},
          open(HERE/"e3_ext_tpr_results.json","w"), indent=2)
for s in SEV:
    d=summary[f"sev{int(s*100)}"]
    print(f"sev {int(s*100)}%: AUC={d['auc']:.4f} TPR@5%FPR={d['tpr05']:.3f} loc_err={d['loc_err_pct']:.1f}%", flush=True)
print("by position, TPR@5%FPR:", {p:{s:round(v['tpr05'],3) for s,v in d.items()} for p,d in by_pos.items()}, flush=True)
