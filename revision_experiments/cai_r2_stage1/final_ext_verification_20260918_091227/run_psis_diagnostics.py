import csv, json, re, sys
from pathlib import Path
import numpy as np, arviz as az
HERE=Path(r'C:\neutra-hmc-surrogate\surrogate_training\revision_experiments\cai_r2_stage1\final_ext_verification_20260918_091227')
S1=Path(r'C:\neutra-hmc-surrogate\surrogate_training\revision_experiments\cai_r2_stage1\run_20260917_094722_21d144f3db844a6b8a2e8167c5282dbe')
def calc(lw):
    lw=np.asarray(lw,float)
    if np.isnan(lw).any() or np.isposinf(lw).any() or not np.isfinite(lw).any(): raise RuntimeError('invalid log weights')
    m=np.max(lw[np.isfinite(lw)]); w=np.zeros_like(lw); ok=np.isfinite(lw); w[ok]=np.exp(lw[ok]-m); w/=w.sum()
    raw=float(1/(w*w).sum()/len(w))
    slw,k=az.psislw(lw,reff=1.0,normalize=True); sw=np.exp(slw)
    return raw,float(1/(sw*sw).sum()/len(sw)),float(k),float(w.max()),float(sw.max())
rows=[]
pat=re.compile(r'(.+)_seed(\d+)_N(2000|8000)\.npz$')
for p in sorted((S1/'routing_arrays').glob('*.npz')):
    m=pat.match(p.name)
    if not m: continue
    a=np.load(p); raw,psess,k,mr,ms=calc(a['log_weights'])
    rows.append(dict(source='stage1',observation=m.group(1),seed=int(m.group(2)),N=int(m.group(3)),raw_ess_ratio=raw,psis_ess_ratio=psess,pareto_k=k,raw_max_weight=mr,psis_max_weight=ms,route_raw_tau05='FAST' if raw>=.5 else 'CORRECTION',route_psis_tau05='FAST' if psess>=.5 else 'CORRECTION',file=str(p)))
with (HERE/'PSIS_STAGE1_DETAIL.csv').open('x',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# final-grid weights: exact replay order and labels from routing CSV
a=np.load(HERE/'FINAL_GRID_WEIGHTS.npz'); grid=list(csv.DictReader((HERE/'FINAL_GRID_ROUTING.csv').open(encoding='utf-8')))
assert len(grid)==a['log_weights'].shape[0]
grow=[]
for i,(r,lw) in enumerate(zip(grid,a['log_weights'])):
    raw,psess,k,mr,ms=calc(lw)
    grow.append(dict(index=i,group=r['group'],parameter=r['parameter'],severity=r['severity'],kappa=r['kappa'],rep=r['rep'],raw_ess_ratio=raw,psis_ess_ratio=psess,pareto_k=k,raw_max_weight=mr,psis_max_weight=ms,route_raw_tau05='FAST' if raw>=.5 else 'CORRECTION',route_psis_tau05='FAST' if psess>=.5 else 'CORRECTION'))
with (HERE/'PSIS_FINAL_GRID_DETAIL.csv').open('x',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(grow[0])); w.writeheader(); w.writerows(grow)
# summaries
def summarize(rs, keys):
    out=[]
    groups={}
    for r in rs:
        key=tuple(r[k] for k in keys); groups.setdefault(key,[]).append(r)
    for key,vals in groups.items():
        raw=np.array([v['raw_ess_ratio'] for v in vals]); pe=np.array([v['psis_ess_ratio'] for v in vals]); kk=np.array([v['pareto_k'] for v in vals])
        d={k:v for k,v in zip(keys,key)}
        d.update(n=len(vals),raw_mean=float(raw.mean()),raw_min=float(raw.min()),raw_max=float(raw.max()),psis_mean=float(pe.mean()),psis_min=float(pe.min()),psis_max=float(pe.max()),k_mean=float(kk.mean()),k_min=float(kk.min()),k_max=float(kk.max()),k_gt_07=int((kk>.7).sum()),k_gt_1=int((kk>1).sum()),raw_correction=int((raw<.5).sum()),psis_correction=int((pe<.5).sum()),route_changes=int(sum(v['route_raw_tau05']!=v['route_psis_tau05'] for v in vals)))
        out.append(d)
    return out
s1sum=summarize(rows,['observation','N'])
gridsum=summarize(grow,['group','parameter'])
for name,data in [('PSIS_STAGE1_SUMMARY.csv',s1sum),('PSIS_FINAL_GRID_SUMMARY.csv',gridsum)]:
    with (HERE/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
field=[r for r in rows if r['observation']=='sailboat_field_frozen']
boundary=[r for r in rows if r['observation']=='sailboat_boundary_e1_075']
report={'arviz_version':az.__version__,'stage1_n':len(rows),'final_grid_n':len(grow),'field':field,'boundary':boundary,'stage1_summary':s1sum,'final_grid_summary':gridsum}
(HERE/'PSIS_SUMMARY.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('ARVIZ',az.__version__)
print('FIELD')
for r in field: print(r)
print('STAGE1_SUMMARY')
for r in s1sum: print(r)
print('FINAL_GRID_SUMMARY')
for r in gridsum: print(r)
